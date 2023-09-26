import  torch, os
import  numpy as np
from    MiniImagenet import MiniImagenet
import  scipy.stats
from    torch.utils.data import DataLoader
from    torch.optim import lr_scheduler
import  random, sys, pickle
import  argparse
import torch.nn as nn
import torch.nn.functional as F
import math
from    torch import optim
from    copy import deepcopy

torch.manual_seed(222)
torch.cuda.manual_seed_all(222)
np.random.seed(222)

device = torch.device('cuda')

def convLayer(in_channels, out_channels, keep_prob=0.0):
    """3*3 convolution with padding,ever time call it the output size become half"""
    cnn_seq = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, 1, 1),
        nn.ReLU(True),
        nn.BatchNorm2d(out_channels),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Dropout(keep_prob)
    )
    return cnn_seq

class Encoder(nn.Module):
    def __init__(self, layer_size=32, num_channels=3, keep_prob=0, image_size=84):
        super(Encoder, self).__init__()
        """
        Build a CNN to produce embeddings
        :param layer_size:64(default)
        :param num_channels:
        :param keep_prob:
        :param image_size:
        """
        self.layer1 = convLayer(num_channels, layer_size, keep_prob)
        self.layer2 = convLayer(layer_size, layer_size, keep_prob)
        self.layer3 = convLayer(layer_size, layer_size, keep_prob)
        self.layer4 = convLayer(layer_size, layer_size, keep_prob)
        self.layer5 = convLayer(layer_size, layer_size, keep_prob)
        self.layer6 = convLayer(layer_size, layer_size, keep_prob)

        self.fc = nn.Linear(800, 5)
        finalSize = int(math.floor(image_size / (2 * 2 * 2 * 2)))
        self.outSize = finalSize * finalSize * layer_size
        self.param_list = []

    def forward(self, image_input):
        """
        Use CNN defined above
        :param image_input:
        :return:
        """
        x = self.layer1(image_input)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = x.view(x.size()[0], -1)
        x = self.fc(x)
        return x

def update_weights(model, new_weights):
    i = 0
    for p in model.parameters():
        p.data = new_weights[i]
        i += 1

def weights_init(m):
    if isinstance(m, nn.Conv2d):
        m.weight.data.normal_(0.0, 0.02)
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def train(model, x_spt, y_spt, x_qry, y_qry, update_step, task_num, criterion, update_lr, optimizer):
    task_num, setsz, c_, h, w = x_spt.size()
    querysz = x_qry.size(1)

    losses_q = [0 for _ in range(update_step + 1)]  # losses_q[i] is the loss on step i
    corrects = [0 for _ in range(update_step + 1)]

    for i in range(task_num):  # inner loop
        weights_init(model)
        logits = model(x_spt[i])

        loss = criterion(logits, y_spt[i])
        grad = torch.autograd.grad(loss, list(model.parameters()))
        fast_weights = list(map(lambda p: p[1] - update_lr * p[0], zip(grad, list(model.parameters()))))

        with torch.no_grad():
            logits_q = model(x_qry[i])

            loss_q = criterion(logits_q, y_qry[i])
            losses_q[0] += loss_q

            pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
            correct = torch.eq(pred_q, y_qry[i]).sum().item()
            corrects[0] = corrects[0] + correct

        update_weights(model, fast_weights)

        with torch.no_grad():
            logits_q = model(x_qry[i])

            loss_q = criterion(logits_q, y_qry[i])
            losses_q[1] += loss_q

            pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
            correct = torch.eq(pred_q, y_qry[i]).sum().item()
            corrects[1] = corrects[1] + correct

        for k in range(1, update_step):
            logits = model(x_spt[i])

            loss = F.cross_entropy(logits, y_spt[i])
            # 2. compute grad on theta_pi
            grad = torch.autograd.grad(loss, list(model.parameters()))
            # 3. theta_pi = theta_pi - train_lr * grad
            fast_weights = list(map(lambda p: p[1] - update_lr * p[0], zip(grad, list(model.parameters()))))
            update_weights(model, fast_weights)

            logits_q = model(x_qry[i])
            # loss_q will be overwritten and just keep the loss_q on last update step.
            loss_q = criterion(logits_q, y_qry[i])
            losses_q[k + 1] += loss_q

            with torch.no_grad():
                pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                correct = torch.eq(pred_q, y_qry[i]).sum().item()  # convert to numpy
                corrects[k + 1] = corrects[k + 1] + correct

    # end of all tasks
    # sum over all losses on query set across all tasks
    loss_q = losses_q[-1] / task_num
    # optimize theta parameters
    optimizer.zero_grad()
    loss_q.backward()
    # print('meta update')
    # for p in self.net.parameters()[:5]:
    # 	print(torch.norm(p).item())
    optimizer.step()

    accs = np.array(corrects) / (querysz * task_num)

    return accs

def test(model, x_spt, y_spt, x_qry, y_qry, update_test_step, task_num, criterion, update_lr, optimizer):
    corrects = [0 for _ in range(update_test_step + 1)]

    querysz = x_qry.size(0)

    net = deepcopy(model)

    logits = net(x_spt)
    loss = criterion(logits, y_spt)

    net.parameters_list()
    grad = torch.autograd.grad(loss, net.parameters_list())
    fast_weights = list(map(lambda p: p[1] - update_lr * p[0], zip(grad, net.parameters_list())))

    with torch.no_grad():
        logits_q = net(x_qry)

        pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
        correct = torch.eq(pred_q, y_qry).sum().item()
        corrects[0] = corrects[0] + correct

    with torch.no_grad():
        update_weights(net, fast_weights)
        model.parameters_list()

        logits_q = net(x_qry)

        pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
        correct = torch.eq(pred_q, y_qry).sum().item()
        corrects[1] = corrects[1] + correct

    for k in range(1, update_test_step):
        logits = net(x_spt)
        loss = F.cross_entropy(logits, y_spt)
        # 2. compute grad on theta_pi
        grad = torch.autograd.grad(loss, net.parameters_list())
        # 3. theta_pi = theta_pi - train_lr * grad
        fast_weights = list(map(lambda p: p[1] - update_lr * p[0], zip(grad, net.parameters_list())))
        update_weights(net, fast_weights)
        model.parameters_list()

        logits_q = net(x_qry)

        with torch.no_grad():
            pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
            correct = torch.eq(pred_q, y_qry).sum().item()  # convert to numpy
            corrects[k + 1] = corrects[k + 1] + correct

    del net

    accs = np.array(corrects) / querysz

    return accs

def main():
    n_way = 5
    k_spt = 1
    k_qry = 15
    imgsz = 84
    task_num = 4
    epoch = 60000
    update_lr = 0.01
    update_step = 5
    update_test_step = 15
    meta_lr = 1e-3

    criterion = nn.CrossEntropyLoss()
    model = Encoder().to(device)

    mini = MiniImagenet('miniimagenet/', mode='train', n_way=n_way, k_shot=k_spt,
                        k_query=k_qry,
                        batchsz=10000, resize=imgsz)
    mini_test = MiniImagenet('miniimagenet/', mode='test', n_way=n_way, k_shot=k_spt,
                             k_query=k_qry,
                             batchsz=100, resize=imgsz)

    optimizer = optim.Adam(model.parameters(), lr=meta_lr)

    for epoch in range(epoch//10000):
        db = DataLoader(mini, task_num, shuffle=True, num_workers=1, pin_memory=True)

        for step, (x_spt, y_spt, x_qry, y_qry) in enumerate(db): #Outer loop
            x_spt, y_spt, x_qry, y_qry = x_spt.to(device), y_spt.to(device), x_qry.to(device), y_qry.to(device)

            model = Encoder().to(device)
            accs = train(model, x_spt, y_spt, x_qry, y_qry, update_step, task_num, criterion, update_lr, optimizer)

            if step % 30 == 0:
                print('step:', step, '\ttraining acc:', accs)

            # if step % 500 == 0:
            #     db_test = DataLoader(mini_test, 1, shuffle=True, num_workers=1, pin_memory=True)
            #     accs_all_test = []
            #
            #     for x_spt, y_spt, x_qry, y_qry in db_test:
            #         x_spt, y_spt, x_qry, y_qry = x_spt.squeeze(0).to(device), y_spt.squeeze(0).to(device), \
            #                                      x_qry.squeeze(0).to(device), y_qry.squeeze(0).to(device)
            #
            #         accs = test(model, x_spt, y_spt, x_qry, y_qry, update_test_step, task_num, criterion, update_lr, optimizer)
            #         accs_all_test.append(accs)
            #
            #     # [b, update_step+1]
            #     accs = np.array(accs_all_test).mean(axis=0).astype(np.float16)
            #     print('Test acc:', accs)

if __name__ == '__main__' :
    main()