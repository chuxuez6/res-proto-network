import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchsummary import summary
from torch.autograd import Variable

from protonets.models import register_model
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from torchsummary import summary

from .utils import euclidean_dist
#from scripts.predict.few_shot import test_single
class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)
sum=torch.zeros(8, 8)
global xs, z_proto, test_hat
class Protonet(nn.Module):
    y_true=[]
    y_pred=[]
    cm=[]

    def __init__(self, encoder):
        super(Protonet, self).__init__()
        self.encoder = encoder
    def loss(self, sample):
        global sum
        global xs, z_proto, test_hat
        xs = Variable(sample['xs']) # support
        xq = Variable(sample['xq']) # query
        n_class = xs.size(0)
        assert xq.size(0) == n_class
        n_support = xs.size(1)
        n_query = xq.size(1)
        target_inds = torch.arange(0, n_class).view(n_class, 1, 1).expand(n_class, n_query, 1).long()
        target_inds = Variable(target_inds, requires_grad=False)
        if xq.is_cuda:
            target_inds = target_inds.cuda()
#将支撑集的向量和查询集向量拼接融合
        x = torch.cat([xs.view(n_class * n_support, *xs.size()[2:]),
                       xq.view(n_class * n_query, *xq.size()[2:])], 0)
#送入特征提取网络提取出特征向量矩阵
        z = self.encoder.forward(x)

#        test_sin = self.encoder.forward(test_single.test_img) #单张图片提取特征
#        print(test_sin)
        z_dim = z.size(-1)
        z_proto = z[:n_class*n_support].view(n_class, n_support, z_dim).mean(1) #计算类原型
        zq = z[n_class*n_support:] #得到查询集的特征向量
        dists = euclidean_dist(zq, z_proto) #度量比较

        #单张测试的度量判断分类
#        dists_test = euclidean_dist(test_sin, z_proto)
#        print(dists_test)
#        log_p_y_test = F.log_softmax(-dists_test, dim=1).view(1, 1, -1)
#        print(log_p_y_test)
#        _, test_hat = log_p_y_test.max(-1)
#        test_hat = test_hat.cpu().numpy()
#        test_hat = test_hat.reshape(-1)
#        print(test_hat)
        #
        log_p_y = F.log_softmax(-dists, dim=1).view(n_class, n_query, -1) #激活函数
        loss_val = -log_p_y.gather(2, target_inds).squeeze().view(-1).mean()
        _, y_hat = log_p_y.max(2) #选取概率最大值
        y_pred=y_hat.cpu().numpy()
        y_pred=y_pred.reshape(-1)
        y_true=target_inds.squeeze().cpu().numpy()
        y_true=y_true.reshape(-1) #处理成合适的向量
        cm = confusion_matrix(y_true, y_pred) #计算混淆矩阵
        sum = sum + cm #将混淆矩阵累加
#        print(sum)
#        plot_confusion_matrix(cm, 'confusion_matrix.png', title='confusion matrix')
        acc_val = torch.eq(y_hat, target_inds.squeeze()).float().mean()
        return loss_val, {
            'loss': loss_val.item(),
            'acc': acc_val.item()
        }

#残差块设计
class Basicblock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Basicblock, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self,x):
        residual = x
        res_out = self.conv(x)
        res_out = self.bn(res_out)
        res_out = self.relu(res_out)
        res_out = self.conv(res_out)
        res_out = self.bn(res_out)
        res_out = self.relu(res_out)
        res_out = self.conv(res_out)
        res_out = self.bn(res_out)
        res_out = self.relu(res_out)
        res_out = res_out+residual
        return res_out

@register_model('protonet_conv')
def load_protonet_conv(**kwargs):
    x_dim = kwargs['x_dim']
    hid_dim = kwargs['hid_dim']
    z_dim = kwargs['z_dim']

    def conv_block(in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
    def conv_block1(in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3,stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
#特征提取网络
    encoder1 = nn.Sequential(
        conv_block(x_dim[0], hid_dim),
        conv_block(hid_dim, hid_dim),
        conv_block1(hid_dim, hid_dim),
        Basicblock(hid_dim, hid_dim),
        Basicblock(hid_dim, hid_dim),
        Basicblock(hid_dim, z_dim),
        nn.AdaptiveAvgPool2d((1, 1)),
        Flatten()
    )
    encoder_outer = encoder1
#    encoder1 = nn.Sequential(
#        conv_block(x_dim[0], hid_dim),
#       conv_block(hid_dim, hid_dim),
#        conv_block(hid_dim, hid_dim),
#        conv_block(hid_dim, z_dim),
#        Flatten()
#    )
#    summary(encoder, (1, 112, 112), batch_size=1, device="cpu")
    summary(encoder1, (1, 56, 56), batch_size=1, device="cpu")
    return Protonet(encoder1)
    # summary(encoder1, (1, 56, 56), batch_size=1, device="cpu")
