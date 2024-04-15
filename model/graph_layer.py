import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from torch.autograd import Variable


# 图卷积网络
class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True, init='xavier'):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        if init == 'uniform':
            #print("| Uniform Initialization")
            self.reset_parameters_uniform()
        elif init == 'xavier':
            #print("| Xavier Initialization")
            self.reset_parameters_xavier()
        elif init == 'kaiming':
            #print("| Kaiming Initialization")
            self.reset_parameters_kaiming()
        else:
            raise NotImplementedError

    def reset_parameters_uniform(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def reset_parameters_xavier(self):
        nn.init.xavier_normal_(self.weight.data, gain=0.02) # Implement Xavier Uniform
        if self.bias is not None:
            nn.init.constant_(self.bias.data, 0.0)

    def reset_parameters_kaiming(self):
        nn.init.kaiming_normal_(self.weight.data, a=0, mode='fan_in')
        if self.bias is not None:
            nn.init.constant_(self.bias.data, 0.0)

    def forward(self, input, adj,b):
        support = torch.bmm(input, self.weight.repeat(b,1,1))
        output = torch.bmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'




# in_features为输入向量的维度，输入向量-》[n,in_features]，n为数量
# out_features为输出向量的维度，输出向量-》[n,out_features]，n为数量
# 例子
# layer=AttentionLayer(in_features=256,out_features=256)
# feature_map -》 [b,c,h,w] 先view为 feature_map=feature_map.permute(0,2,3,1).view(-1,c) -》[b*h*w,in_features] in_features=c为维度
# out_feature=layer(feature_map) -》[b*h*w,out_features]
class AttentionLayer(nn.Module):
    """
    Simple Attention layer, huihuihui
    """

    def __init__(self, in_features, out_features, bias=False, init='xavier'):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight_k = Parameter(torch.FloatTensor(in_features, out_features))
        self.weight_q = Parameter(torch.FloatTensor(in_features, out_features))
        self.weight_v = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        if init == 'uniform':
            #print("| Uniform Initialization")
            self.reset_parameters_uniform()
        elif init == 'xavier':
            #print("| Xavier Initialization")
            self.reset_parameters_xavier()
        elif init == 'kaiming':
            #print("| Kaiming Initialization")
            self.reset_parameters_kaiming()
        else:
            raise NotImplementedError

    def reset_parameters_uniform(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def reset_parameters_xavier(self):
        nn.init.xavier_normal_(self.weight.data, gain=0.02) # Implement Xavier Uniform
        if self.bias is not None:
            nn.init.constant_(self.bias.data, 0.0)

    def reset_parameters_kaiming(self):
        nn.init.kaiming_normal_(self.weight.data, a=0, mode='fan_in')
        if self.bias is not None:
            nn.init.constant_(self.bias.data, 0.0)

    def forward(self, input):
        k = torch.mm(input, self.weight_k)
        q = torch.mm(input, self.weight_q)
        v = torch.mm(input, self.weight_v)
        norm_k = torch.norm(k, 2, 1).view(-1, 1)
        norm_q = torch.norm(q, 2, 1).view(-1, 1)
        sim = torch.div(torch.mm(k, q.t()), torch.mm(norm_k, norm_q.t()))
        output = torch.mm(torch.softmax(sim, dim=1), v)


        # 不需要bias
        # if self.bias is not None:
        #     return output + self.bias
        # else:
        #     return output

        return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

# 图注意力网络，取消了参数的学习？？？ 来自Attention-Based Graph Neural Network For Semi-supervised Learning
class GraphAttentionParameterLayer(nn.Module):

    def __init__(self, requires_grad=True):
        super(GraphAttentionParameterLayer, self).__init__()
        if requires_grad:
            # unifrom initialization
            self.beta = Parameter(torch.Tensor(1).uniform_(
                0, 1), requires_grad=requires_grad)
        else:
            self.beta = Variable(torch.zeros(1), requires_grad=requires_grad)

    def forward(self, x, adj, aff_cropping):
        # ？？？使用project还是classification，project [-1,1] classification >0

        neighbor = torch.mm(x, x.t()).detach()
        neighbor[aff_cropping == 0] = 0
        neighbor = self.beta * neighbor
        masked = 10 * adj + neighbor

        # propagation matrix
        P = F.softmax(masked, dim=1)

        # attention-guided propagation
        output = torch.mm(P, x)
        return output

    def __repr__(self):
        return self.__class__.__name__ + ' (16 -> 16)'


class GraphAttentionLayer(nn.Module):

    def __init__(self, tau=10):
        super(GraphAttentionLayer, self).__init__()
        self.tau = tau

    def forward(self, x, adj):
        # ？？？使用project还是classification，project [-1,1] classification >0

        # neighbor = torch.mm(x, x.t()).detach()
        # neighbor[aff_cropping == 0] = 0
        # neighbor = self.beta * neighbor
        # masked = 10 * adj + neighbor

        masked = self.tau * adj

        # propagation matrix
        P = F.softmax(masked, dim=1)

        # attention-guided propagation
        output = torch.mm(P, x)
        return output

    def __repr__(self):
        return self.__class__.__name__ + ' (16 -> 16)'