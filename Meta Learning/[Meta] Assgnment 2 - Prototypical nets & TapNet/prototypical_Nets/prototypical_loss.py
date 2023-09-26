import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.modules import Module

def euclidean_dist(x, y):
    # x: N x D
    # y: M x D
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    assert d == y.size(1)

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    return torch.pow(x - y, 2).sum(2)

class PrototypicalLoss(Module):
    '''
    Loss class deriving from Module for the prototypical loss function defined below
    '''

    def __init__(self):
        super(PrototypicalLoss, self).__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def forward(self, input, target, n_support):
        return prototypical_loss(input, target, n_support, self.device)


def prototypical_loss(input, target, n_support, device):
    '''
    Inspired by https://github.com/jakesnell/prototypical-networks/blob/master/protonets/models/few_shot.py

    Compute the barycentres by averaging the features of n_support
    samples for each class in target, computes then the distances from each
    samples' features to each one of the barycentres, computes the
    log_probability for each n_query samples for each one of the current
    classes, of appartaining to a class c, loss and accuracy are then computed
    and returned
    Args:
    - input: the model output for a batch of samples
    - target: ground truth for the above batch of samples
    - n_support: number of samples to keep in account when computing
      barycentres, for each one of the current classes
    '''

    classes = torch.unique(target)
    n_classes = len(classes)

    # Make prototypes
    support_idxs = torch.stack(list(map(lambda c: target.eq(c).nonzero()[:n_support].squeeze(1), classes)))
    prototypes = torch.stack([input[idx_list].mean(0) for idx_list in support_idxs])

    # Make query samples
    n_query = target.eq(classes[0].item()).sum().item() - n_support
    query_idxs = torch.stack(list(map(lambda c: target.eq(c).nonzero()[n_support:], classes))).view(-1)
    query_samples = input[query_idxs]

    feature_dims = 1600  # 1600 for miniimagenet, 64 for omniglot
    learnable_scale = nn.Parameter(torch.FloatTensor(1).fill_(1.0), requires_grad=True)
    dists = learnable_scale * euclidean_dist(query_samples, prototypes) / feature_dims

    log_p_y = F.log_softmax(-dists, dim=1)
    y_hat = log_p_y.argmax(1)
    target_label = torch.arange(0, n_classes, 1 / n_query).long().to(device)

    loss_val = torch.nn.NLLLoss()(log_p_y, target_label)
    acc_val = y_hat.eq(target_label.squeeze()).float().mean()

    return loss_val, acc_val
