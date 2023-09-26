import torch.nn as nn
import torch
from utils.conv_block import ConvBlock
from torch.nn import functional as F

#Calculating the distance
def euclidean_dist(x, y):
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    assert d == y.size(1)

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    return torch.pow(x - y, 2).sum(2)

#Prototypical_Net
class ProtoNet(nn.Module):
    def __init__(self, input_dim, hid_dim=64, z_dim=64):
        super(ProtoNet, self).__init__()
        self.block1 = ConvBlock(input_dim, hid_dim, 3, max_pool=2, padding=1)
        self.block2 = ConvBlock(hid_dim, hid_dim, 3, max_pool=2, padding=1)
        self.block3 = ConvBlock(hid_dim, hid_dim, 3, max_pool=2, padding=1)
        self.block4 = ConvBlock(hid_dim, z_dim, 3, max_pool=2, padding=1)

    def forward(self, x):
        out = self.block1(x)
        out = self.block2(out)
        out = self.block3(out)
        out = self.block4(out)
        out = out.view(out.size(0), -1)

        return out

    def prototypical_loss(self, input, target, n_support, device):
        classes = torch.unique(target)
        n_classes = len(classes)

        # Make prototypes
        support_idxs = torch.stack(list(map(lambda c: target.eq(c).nonzero()[:n_support].squeeze(1), classes)))
        prototypes = torch.stack([input[idx_list].mean(0) for idx_list in support_idxs])

        # Make query samples
        n_query = target.eq(classes[0].item()).sum().item() - n_support
        query_idxs = torch.stack(list(map(lambda c: target.eq(c).nonzero()[n_support:], classes))).view(-1)
        query_samples = input[query_idxs]

        # Scaling euclidean
        feature_dims = 1600
        learnable_scale = nn.Parameter(torch.FloatTensor(1).fill_(1.0), requires_grad=True).to(device)
        dists = learnable_scale * euclidean_dist(query_samples, prototypes) / feature_dims

        log_p_y = F.log_softmax(-dists, dim=1)
        y_hat = log_p_y.argmax(1)
        target_label = torch.arange(0, n_classes, 1 / n_query).long().to(device)

        loss_val = torch.nn.NLLLoss()(log_p_y, target_label)
        acc_val = y_hat.eq(target_label.squeeze()).float().mean()

        return loss_val, acc_val
