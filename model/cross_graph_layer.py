import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from torch.autograd import Variable


from model.graph_layer import *

class CrossGraph(nn.Module):

    def __init__(self, num_classess,rep_dim):
        super(CrossGraph, self).__init__()
        self.num_classes=num_classess
        self.rep_dim=rep_dim
        self.graph_c=GraphConvolution(self.num_classes, self.num_classes)
        self.graph_p=GraphConvolution(self.rep_dim,self.rep_dim)

    def cosine_similarity(self, x):
        x_relu = nn.ReLU(inplace=True)(x)
        x_norm = x_relu / (torch.norm(x_relu, dim=2, keepdim=True) + 10e-10)
        dist = x_norm.bmm(x_norm.permute(0, 2, 1))
        return dist

    def forward(self, rep,pred,relu=False):
        with torch.no_grad():
            b, h, w = rep.size(0), rep.size(2), rep.size(3)
            rep_batch_temp = rep.permute(0, 2, 3, 1).reshape(b, -1, self.rep_dim)
            pred_batch_temp = pred.permute(0, 2, 3, 1).reshape(b, -1, self.num_classes)

            adj = self.cosine_similarity(rep_batch_temp)
            adj_c = torch.softmax(adj / 0.1, dim=2)

            pred_batch_temp = self.graph_c(pred_batch_temp, adj_c,b)


            pred_batch_temp_psl = torch.argmax(pred_batch_temp, dim=2).detach()
            pred_batch_temp_psl = F.one_hot(pred_batch_temp_psl, num_classes=self.num_classes).float()
            adj_p = adj_c.clone()
            adj_p=adj_p*self.cosine_similarity(pred_batch_temp_psl)
            adj_p = F.normalize(adj_p, p=1, dim=2)

            rep_batch_temp = self.graph_p(rep_batch_temp, adj_p,b)

            if relu:
                pred_batch_temp = F.relu(pred_batch_temp, inplace=True)
                rep_batch_temp = F.relu(rep_batch_temp, inplace=True)

            rep_graph = rep_batch_temp.permute(0, 2, 1).view(b, self.rep_dim, h, w)
            pred_graph = pred_batch_temp.permute(0, 2, 1).view(b, self.num_classes, h, w)


        return rep_graph,pred_graph

    def __repr__(self):
        return self.__class__.__name__ + ' (16 -> 16)'