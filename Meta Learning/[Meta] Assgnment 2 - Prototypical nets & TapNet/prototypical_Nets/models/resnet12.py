# Reimpliment chainer model to pythorch https://github.com/istarjun/TapNet

import torch
import torch.nn as nn
import torch.nn.functional as F

def euclidean_dist(x, y):
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    assert d == y.size(1)

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    return torch.pow(x - y, 2).sum(2)

class ResNet_12(nn.Module):
    def __init__(self):
        super(ResNet_12, self).__init__()

        self.l_conv1_1 = nn.Conv2d(3, 64, (3, 3), padding=1)
        self.l_norm1_1 = nn.BatchNorm2d(64)
        self.l_conv1_2 = nn.Conv2d(64, 64, (3, 3), padding=1)
        self.l_norm1_2 = nn.BatchNorm2d(64)
        self.l_conv1_3 = nn.Conv2d(64, 64, (3, 3), padding=1)
        self.l_norm1_3 = nn.BatchNorm2d(64)
        self.l_conv1_r = nn.Conv2d(3, 64, (3, 3), padding=1)
        self.l_norm1_r = nn.BatchNorm2d(64)

        self.l_conv2_1 = nn.Conv2d(64, 128, (3, 3), padding=1)
        self.l_norm2_1 = nn.BatchNorm2d(128)
        self.l_conv2_2 = nn.Conv2d(128, 128, (3, 3), padding=1)
        self.l_norm2_2 = nn.BatchNorm2d(128)
        self.l_conv2_3 = nn.Conv2d(128, 128, (3, 3), padding=1)
        self.l_norm2_3 = nn.BatchNorm2d(128)
        self.l_conv2_r = nn.Conv2d(64, 128, (3, 3), padding=1)
        self.l_norm2_r = nn.BatchNorm2d(128)

        self.l_conv3_1 = nn.Conv2d(128, 256, (3, 3), padding=1)
        self.l_norm3_1 = nn.BatchNorm2d(256)
        self.l_conv3_2 = nn.Conv2d(256, 256, (3, 3), padding=1)
        self.l_norm3_2 = nn.BatchNorm2d(256)
        self.l_conv3_3 = nn.Conv2d(256, 256, (3, 3), padding=1)
        self.l_norm3_3 = nn.BatchNorm2d(256)
        self.l_conv3_r = nn.Conv2d(128, 256, (3, 3), padding=1)
        self.l_norm3_r = nn.BatchNorm2d(256)

        self.l_conv4_1 = nn.Conv2d(256, 512, (3, 3), padding=1)
        self.l_norm4_1 = nn.BatchNorm2d(512)
        self.l_conv4_2 = nn.Conv2d(512, 512, (3, 3), padding=1)
        self.l_norm4_2 = nn.BatchNorm2d(512)
        self.l_conv4_3 = nn.Conv2d(512, 512, (3, 3), padding=1)
        self.l_norm4_3 = nn.BatchNorm2d(512)
        self.l_conv4_r = nn.Conv2d(256, 512, (3, 3), padding=1)
        self.l_norm4_r = nn.BatchNorm2d(512)

    def forward(self, x, batchsize, train=True):
        x2 = torch.reshape(x, (batchsize, 84, 84, 3))
        x3 = x2.permute((0, 3, 1, 2))

        c1_r = self.l_conv1_r(x3)
        n1_r = self.l_norm1_r(c1_r)

        c1_1 = self.l_conv1_1(x3)
        n1_1 = self.l_norm1_1(c1_1)
        a1_1 = F.relu(n1_1)

        c1_2 = self.l_conv1_2(a1_1)
        n1_2 = self.l_norm1_2(c1_2)
        a1_2 = F.relu(n1_2)

        c1_3 = self.l_conv1_3(a1_2)
        n1_3 = self.l_norm1_3(c1_3)

        a1_3 = F.relu(n1_3 + n1_r)

        p1 = F.max_pool2d(a1_3, 2)
        p1 = F.dropout(p1, 0.3)

        c2_r = self.l_conv2_r(p1)
        n2_r = self.l_norm2_r(c2_r)

        c2_1 = self.l_conv2_1(p1)
        n2_1 = self.l_norm2_1(c2_1)
        a2_1 = F.relu(n2_1)

        c2_2 = self.l_conv2_2(a2_1)
        n2_2 = self.l_norm2_2(c2_2)
        a2_2 = F.relu(n2_2)

        c2_3 = self.l_conv2_3(a2_2)
        n2_3 = self.l_norm2_3(c2_3)

        a2_3 = F.relu(n2_3 + n2_r)

        p2 = F.max_pool2d(a2_3, 2)
        p2 = F.dropout(p2, 0.2)
        c3_r = self.l_conv3_r(p2)
        n3_r = self.l_norm3_r(c3_r)

        c3_1 = self.l_conv3_1(p2)
        n3_1 = self.l_norm3_1(c3_1)
        a3_1 = F.relu(n3_1)

        c3_2 = self.l_conv3_2(a3_1)
        n3_2 = self.l_norm3_2(c3_2)
        a3_2 = F.relu(n3_2)

        c3_3 = self.l_conv3_3(a3_2)
        n3_3 = self.l_norm3_3(c3_3)

        a3_3 = F.relu(n3_3 + n3_r)

        p3 = F.max_pool2d(a3_3, 2)
        p3 = F.dropout(p3, 0.2)

        c4_r = self.l_conv4_r(p3)
        n4_r = self.l_norm4_r(c4_r)

        c4_1 = self.l_conv4_1(p3)
        n4_1 = self.l_norm4_1(c4_1)
        a4_1 = F.relu(n4_1)

        c4_2 = self.l_conv4_2(a4_1)
        n4_2 = self.l_norm4_2(c4_2)
        a4_2 = F.relu(n4_2)

        c4_3 = self.l_conv4_3(a4_2)
        n4_3 = self.l_norm4_3(c4_3)

        a4_3 = F.relu(n4_3 + n4_r)

        p4 = F.max_pool2d(a4_3, 2)
        p4 = F.dropout(p4, 0.2)

        p5 = F.avg_pool2d(p4, 6, padding=1)
        h_t = torch.reshape(p5, (batchsize, -1))

        return h_t

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