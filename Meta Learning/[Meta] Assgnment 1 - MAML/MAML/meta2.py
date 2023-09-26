import  torch
from    torch import nn
from    torch import optim
from    torch.nn import functional as F
from    torch.utils.data import TensorDataset, DataLoader
from    torch import optim
import  numpy as np

from    learner import Learner
from    copy import deepcopy



class Meta2(nn.Module):
    """
    Meta Learner, Finetunes only the classifier weights
    """
    def __init__(self, args, config):
        """

        :param args:
        """
        super(Meta2, self).__init__()

        self.update_lr = args.update_lr
        self.meta_lr = args.meta_lr
        self.n_way = args.n_way
        self.k_spt = args.k_spt
        self.k_qry = args.k_qry
        self.task_num = args.task_num
        self.update_step = args.update_step
        self.update_step_test = args.update_step_test


        self.net = Learner(config, args.imgc, args.imgsz)
        self.meta_optim = optim.Adam(self.net.parameters(), lr=self.meta_lr)




    def clip_grad_by_norm_(self, grad, max_norm):
        """
        in-place gradient clipping.
        :param grad: list of gradients
        :param max_norm: maximum norm allowable
        :return:
        """

        total_norm = 0
        counter = 0
        for g in grad:
            param_norm = g.data.norm(2)
            total_norm += param_norm.item() ** 2
            counter += 1
        total_norm = total_norm ** (1. / 2)

        clip_coef = max_norm / (total_norm + 1e-6)
        if clip_coef < 1:
            for g in grad:
                g.data.mul_(clip_coef)

        return total_norm/counter


    def forward(self, x_spt, y_spt, x_qry, y_qry):
        """

        :param x_spt:   [b, setsz, c_, h, w]
        :param y_spt:   [b, setsz]
        :param x_qry:   [b, querysz, c_, h, w]
        :param y_qry:   [b, querysz]
        :return:
        """
        task_num, setsz, c_, h, w = x_spt.size()
        querysz = x_qry.size(1)

        losses_q = [0 for _ in range(self.update_step + 1)]  # losses_q[i] is the loss on step i
        corrects = [0 for _ in range(self.update_step + 1)]


        for i in range(task_num):

            #Run the i-th task and compute loss for k=0
            logits = self.net(x_spt[i], vars=None, bn_training=True)
            loss = F.cross_entropy(logits, y_spt[i])

            #Gradients of classifier
            grad = torch.autograd.grad(loss, self.net.parameters()[-2:])
            fast_weights = self.net.parameters()
            classifier_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad[-2:], self.net.parameters()[-2:])))
            fast_weights[-1] = nn.Parameter(classifier_weights[-1].detach().clone())
            fast_weights[-2] = nn.Parameter(classifier_weights[-2].detach().clone())

            # this is the loss and accuracy before first update
            with torch.no_grad():
                # [setsz, nway]
                logits_q = self.net(x_qry[i], self.net.parameters(), bn_training=True)
                loss_q = F.cross_entropy(logits_q, y_qry[i])
                losses_q[0] += loss_q

                pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                correct = torch.eq(pred_q, y_qry[i]).sum().item()
                corrects[0] = corrects[0] + correct

            # this is the loss and accuracy after the first update
            with torch.no_grad():
                logits_q = self.net(x_qry[i], fast_weights, bn_training=True)
                loss_q = F.cross_entropy(logits_q, y_qry[i])
                losses_q[1] += loss_q
                # [setsz]
                pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                correct = torch.eq(pred_q, y_qry[i]).sum().item()
                corrects[1] = corrects[1] + correct

            for k in range(1, self.update_step):
                # 1. run the i-th task and compute loss for k=1~K-1
                logits = self.net(x_spt[i], fast_weights, bn_training=True)
                loss = F.cross_entropy(logits, y_spt[i])
                # 2. compute grad on theta_pi

                grad = torch.autograd.grad(loss, fast_weights[-2:])
                # 3. theta_pi = theta_pi - train_lr * grad
                classifier_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad[-2:], fast_weights[-2:])))
                fast_weights[-1] = nn.Parameter(classifier_weights[-1].detach().clone())
                fast_weights[-2] = nn.Parameter(classifier_weights[-2].detach().clone())

                logits_q = self.net(x_qry[i], fast_weights, bn_training=True)
                # loss_q will be overwritten and just keep the loss_q on last update step.
                loss_q = F.cross_entropy(logits_q, y_qry[i])
                losses_q[k + 1] += loss_q

                with torch.no_grad():
                    pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                    correct = torch.eq(pred_q, y_qry[i]).sum().item()  # convert to numpy
                    corrects[k + 1] = corrects[k + 1] + correct



        # Last entry of the list is sum over all losses on query set across all tasks
        loss_q = losses_q[-1] / task_num

        # optimize theta parameters
        self.meta_optim.zero_grad()
        loss_q.backward()

        self.meta_optim.step()


        accs = np.array(corrects) / (querysz * task_num)

        return accs


    def finetunning(self, x_spt, y_spt, x_qry, y_qry):
        """

        :param x_spt:   [setsz, c_, h, w]
        :param y_spt:   [setsz]
        :param x_qry:   [querysz, c_, h, w]
        :param y_qry:   [querysz]
        :return:
        """
        assert len(x_spt.shape) == 4

        querysz = x_qry.size(0)
        corrects = [0 for _ in range(self.update_step_test + 1)]

        # Deepcopying the model
        net = deepcopy(self.net)

        # Run the i-th task and compute loss for k=0
        logits = net(x_spt)
        loss = F.cross_entropy(logits, y_spt)

        #Classifier weights : net.parameters()[-2:]
        grad = torch.autograd.grad(loss, net.parameters()[-2:])
        fast_weights = net.parameters()
        classifier_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad[-2:], net.parameters()[-2:])))
        fast_weights[-1] = nn.Parameter(classifier_weights[-1].detach().clone())
        fast_weights[-2] = nn.Parameter(classifier_weights[-2].detach().clone())

        # this is the loss and accuracy before first update
        with torch.no_grad():
            logits_q = net(x_qry, net.parameters(), bn_training=True)
            pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
            correct = torch.eq(pred_q, y_qry).sum().item()
            corrects[0] = corrects[0] + correct

        # this is the loss and accuracy after the first update
        with torch.no_grad():
            logits_q = net(x_qry, fast_weights, bn_training=True)
            pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
            correct = torch.eq(pred_q, y_qry).sum().item()
            corrects[1] = corrects[1] + correct

        for k in range(1, self.update_step_test):
            # Compute loss with respect to K examples
            logits = net(x_spt, fast_weights, bn_training=True)
            loss = F.cross_entropy(logits, y_spt)
            # Compute gradient on theta_pi
            grad = torch.autograd.grad(loss, fast_weights[-2:])
            # Update weights : theta_pi = theta_pi - train_lr * grad
            classifier_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad[-2:], fast_weights[-2:])))
            fast_weights[-1] = nn.Parameter(classifier_weights[-1].detach().clone())
            fast_weights[-2] = nn.Parameter(classifier_weights[-2].detach().clone())
            logits_q = net(x_qry, fast_weights, bn_training=True)

            with torch.no_grad():
                pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                correct = torch.eq(pred_q, y_qry).sum().item()  # convert to numpy
                corrects[k + 1] = corrects[k + 1] + correct


        del net

        accs = np.array(corrects) / querysz
        return accs




def main():
    pass


if __name__ == '__main__':
    main()
