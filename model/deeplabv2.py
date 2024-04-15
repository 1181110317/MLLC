# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch.nn as nn
import torch.nn.functional as F
import math
import torch.utils.model_zoo as model_zoo
import torch
import numpy as np
from model.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d
from torch.autograd import Variable
from model.cross_graph_layer import *
from model.non_local import NLBlock

affine_par = True


def outS(i):
    i = int(i)
    i = (i + 1) / 2
    i = int(np.ceil((i + 1) / 2.0))
    i = (i + 1) / 2
    return i


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, affine=affine_par)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, affine=affine_par)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None, BatchNorm=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False)  # change
        self.bn1 = BatchNorm(planes, affine=affine_par)
        # for i in self.bn1.parameters():
        #     i.requires_grad = False

        padding = dilation
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,  # change
                               padding=padding, bias=False, dilation=dilation)
        self.bn2 = BatchNorm(planes, affine=affine_par)
        # for i in self.bn2.parameters():
        #     i.requires_grad = False
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = BatchNorm(planes * 4, affine=affine_par)
        # for i in self.bn3.parameters():
        #     i.requires_grad = False
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Classifier_Module(nn.Module):
    def __init__(self, inplanes, dilation_series, padding_series, num_classes):
        super(Classifier_Module, self).__init__()
        self.conv2d_list = nn.ModuleList()
        for dilation, padding in zip(dilation_series, padding_series):
            self.conv2d_list.append(
                nn.Conv2d(inplanes, num_classes, kernel_size=3, stride=1, padding=padding, dilation=dilation, bias=True))

        for m in self.conv2d_list:
            m.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.conv2d_list[0](x)
        for i in range(len(self.conv2d_list) - 1):
            out += self.conv2d_list[i + 1](x)
            return out

class SEBlock(nn.Module):
    def __init__(self, inplanes, r = 16):
        super(SEBlock, self).__init__()
        self.global_pool = nn.AdaptiveAvgPool2d((1,1))
        self.se = nn.Sequential(
                nn.Linear(inplanes, inplanes//r),
                nn.ReLU(inplace=True),
                nn.Linear(inplanes//r, inplanes),
                nn.Sigmoid()
        )
    def forward(self, x):
        xx = self.global_pool(x)
        xx = xx.view(xx.size(0), xx.size(1))
        se_weight = self.se(xx).unsqueeze(-1).unsqueeze(-1)
        return x.mul(se_weight)

class Classifier_Module2(nn.Module):
    def __init__(self, inplanes, dilation_series, padding_series, num_classes, droprate = 0.1, use_se = True):
        super(Classifier_Module2, self).__init__()
        self.conv2d_list = nn.ModuleList()
        self.conv2d_list.append(
                nn.Sequential(*[
                nn.Conv2d(inplanes, 256, kernel_size=1, stride=1, padding=0, dilation=1, bias=True),
                nn.GroupNorm(num_groups=32, num_channels=256, affine = True),
                nn.ReLU(inplace=True) ]))

        for dilation, padding in zip(dilation_series, padding_series):
            #self.conv2d_list.append(
            #    nn.BatchNorm2d(inplanes))
            self.conv2d_list.append(
                nn.Sequential(*[
                #nn.ReflectionPad2d(padding),
                nn.Conv2d(inplanes, 256, kernel_size=3, stride=1, padding=padding, dilation=dilation, bias=True), 
                nn.GroupNorm(num_groups=32, num_channels=256, affine = True),
                nn.ReLU(inplace=True) ]))
 
        if use_se:
            self.bottleneck = nn.Sequential(*[SEBlock(256 * (len(dilation_series) + 1)),
                        nn.Conv2d(256 * (len(dilation_series) + 1), 256, kernel_size=3, stride=1, padding=1, dilation=1, bias=True) ,
                        nn.GroupNorm(num_groups=32, num_channels=256, affine = True) ])
        else:
            self.bottleneck = nn.Sequential(*[
                nn.Conv2d(256 * (len(dilation_series) + 1), 256, kernel_size=3, stride=1, padding=1, dilation=1, bias=True) ,
                nn.GroupNorm(num_groups=32, num_channels=256, affine = True) ])

        self.head = nn.Sequential(*[nn.Dropout2d(droprate),
            nn.Conv2d(256, num_classes, kernel_size=1, padding=0, dilation=1, bias=False) ])

        ##########init#######
        for m in self.conv2d_list:
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.InstanceNorm2d) or isinstance(m, nn.GroupNorm):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        for m in self.bottleneck:
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.InstanceNorm2d) or isinstance(m, nn.GroupNorm) or isinstance(m, nn.LayerNorm):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        for m in self.head:
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.001)

    def forward(self, x, get_feat=False):
        out = self.conv2d_list[0](x)
        for i in range(len(self.conv2d_list) - 1):
            out = torch.cat( (out, self.conv2d_list[i+1](x)), 1)
        out = self.bottleneck(out)
        if get_feat:
            out_dict = {}
            out = self.head[0](out)
            out_dict['feat'] = out
            out = self.head[1](out)
            out_dict['out'] = out
            return out_dict
        else:
            out = self.head(out)
            return out


class G2Net(nn.Module):
    def __init__(self, num_classes, BatchNorm):
        super(G2Net, self).__init__()

        self.class_graph_conv2d = nn.Sequential(*[
            # nn.ReflectionPad2d(padding),
            nn.Conv2d(num_classes, num_classes, kernel_size=1, bias=True),
            #BatchNorm(num_classes, affine=affine_par),
            #nn.ReLU(inplace=True),
            nn.Conv2d(num_classes, num_classes, kernel_size=1, bias=True)])


        self.pro_graph_conv2d = nn.Sequential(*[
            # nn.ReflectionPad2d(padding),
            nn.Conv2d(256, 256, kernel_size=1, bias=True),
            #BatchNorm(256, affine=affine_par),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=1, bias=True)])

        self.num_classes=num_classes

        for m in self.class_graph_conv2d:
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.InstanceNorm2d) or isinstance(m, nn.GroupNorm) or isinstance(m, nn.LayerNorm):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        for m in self.pro_graph_conv2d:
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.InstanceNorm2d) or isinstance(m, nn.GroupNorm) or isinstance(m, nn.LayerNorm):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


    def cosine_similarity(self, x_relu):
        #x_relu = nn.ReLU(inplace=True)(x)
        x_norm = x_relu / (torch.norm(x_relu, dim=2, keepdim=True) + 10e-10)
        dist = x_norm.bmm(x_norm.permute(0, 2, 1))
        #去除负相关变量
        dist[dist<0]=0
        return dist

    def proportion_class_weight(self,prob,max,proportion):
        #num_pixels=max.size(0)*max.size(1)*max.size(2)
        if proportion==0.0:
            return (prob!=prob)
        mask=(prob>=0.95)
        for i in range(self.num_classes):
            class_i_prob=prob[max==i]
            if class_i_prob.size(0)==0:
                continue
            threshold=torch.sort(class_i_prob,descending=True)[0][int(class_i_prob.size(0)*proportion)]
            mask[(prob>=threshold)&(max==i)]=True

        return mask


    def g_class_weight(self,prob,max,class_weight,threshold=0.95):
        class_weight_mask=(class_weight/(torch.max(class_weight)+1e-10))*threshold
        mask=(prob>=threshold)
        for i in range(self.num_classes):
            mask[(prob>=class_weight_mask[i])&(max==i)]=True
        return mask

    # def g_class_weight(self,prob,max,class_weight=None,threshold=0.95):
    #     class_every_co=torch.zeros(self.num_classes).cuda()
    #     mask_threshold=(prob>=threshold)
    #     for i in range(self.num_classes):
    #         class_every_co[i]=(mask_threshold&(max==i)).sum()
    #     class_weight=(class_every_co/(torch.max(class_every_co)+1e-10))*threshold
    #     mask=(prob>=threshold)
    #     for i in range(self.num_classes):
    #         mask[(prob>=class_weight[i])&(max==i)]=True
    #     return mask

    def forward(self, pred,feat,class_weight=None,proportion=0.0):
        if class_weight==None:
            class_weight=torch.tensor([1 for i in range(self.num_classes)]).cuda()

        batch_size, c, h, w = feat.shape

        feat = feat.view(batch_size, 256, -1).permute(0, 2, 1).contiguous()


        with torch.no_grad():
            pred1_prob, pred1_max = torch.max(torch.softmax(pred, dim=1), dim=1)
            mask = self.g_class_weight(pred1_prob, pred1_max,class_weight).float().view(batch_size, -1).unsqueeze(-1).repeat(1, 1,self.num_classes).float()
            aff1 = self.cosine_similarity(feat)
            aff1[aff1 < torch.sort(aff1, descending=True, dim=-1)[0][:,:, 20].unsqueeze(-1).repeat(1, 1, h * w)] = 0
            aff1 = aff1 + aff1.permute(0, 2, 1)
            # aff1.diagonal(0,2).fill_(0)

        pred = pred.view(batch_size, self.num_classes, -1).permute(0, 2, 1).contiguous()
        init_pred1 = pred.clone()
        diag_aff1_pred = torch.diag_embed(torch.pow(torch.sum(aff1, dim=-1)+1e-10, -0.5))
        aff1_pred = torch.matmul(diag_aff1_pred, aff1)
        aff1_pred = torch.matmul(aff1_pred, diag_aff1_pred)
        for l in range(1):
            # pred1=0.8*torch.matmul(aff1_pred,pred1)+0.2*init_pred1
            pred = torch.matmul(aff1_pred, pred)
        pred = (0.8 * pred + 0.2 * init_pred1) * (1 - mask) + (0.2 * pred + 0.8 * init_pred1) * mask
        #pred=(pred+init_pred1)/2
        pred = pred.permute(0, 2, 1).contiguous().view(batch_size,self.num_classes,h,w)
        pred=self.class_graph_conv2d(pred)

        with torch.no_grad():
            pred1_one_hot = F.one_hot(pred1_max, num_classes=self.num_classes).float().permute(0, 2, 3,1).contiguous().view(batch_size, -1, self.num_classes).detach()
            aff1_feat=self.cosine_similarity(torch.softmax(pred,dim=1).view(batch_size,self.num_classes,-1).permute(0,2,1).contiguous())
            aff1_feat[self.cosine_similarity(pred1_one_hot) == 0] = 0
            #aff1_feat=aff1_feat*aff1
            diag_aff1_feat = torch.diag_embed(torch.pow(torch.sum(aff1_feat, dim=-1) + 1e-10, -0.5))
            aff1_feat = torch.matmul(diag_aff1_feat, aff1_feat)
            aff1_feat = torch.matmul(aff1_feat, diag_aff1_feat)

        new_feat=torch.matmul(aff1_feat, feat).permute(0, 2, 1).contiguous().view(batch_size,c,h,w)
        #feat=feat.permute(0, 2, 1).contiguous().view(batch_size,c,h,w)

        feat=self.pro_graph_conv2d(new_feat)
        #feat=F.normalize(feat,dim=1)

        return pred,feat






class ResNet101(nn.Module):
    def __init__(self, block, layers, num_classes, BatchNorm, bn_clr=False):
        self.inplanes = 64
        self.bn_clr = bn_clr
        super(ResNet101, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = BatchNorm(64, affine=affine_par)
        # for i in self.bn1.parameters():
        #     i.requires_grad = False
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=True)  # change
        self.layer1 = self._make_layer(block, 64, layers[0], BatchNorm=BatchNorm)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, BatchNorm=BatchNorm)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=1, dilation=2, BatchNorm=BatchNorm)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilation=4, BatchNorm=BatchNorm)
        #self.layer5 = self._make_pred_layer(Classifier_Module, 2048, [6, 12, 18, 24], [6, 12, 18, 24], num_classes)
        self.layer5 = self._make_pred_layer(Classifier_Module2, 2048, [6, 12, 18, 24], [6, 12, 18, 24], num_classes)
        self.layer6 = self._make_pred_layer(Classifier_Module2, 2048, [6, 12, 18, 24], [6, 12, 18, 24], 256)

        self.g2layer1=G2Net(num_classes,BatchNorm)
        self.g2layer2 = G2Net(num_classes, BatchNorm)
        self.num_classes=num_classes
        self.feat_d=256

        if self.bn_clr:
            self.bn_pretrain = BatchNorm(2048, affine=affine_par)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, 0.01)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, SynchronizedBatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
                #        for i in m.parameters():
                #            i.requires_grad = False

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1, BatchNorm=None):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion or dilation == 2 or dilation == 4:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                BatchNorm(planes * block.expansion, affine=affine_par))
        # for i in downsample._modules['1'].parameters():
        #     i.requires_grad = False
        layers = []
        layers.append(block(self.inplanes, planes, stride, dilation=dilation, downsample=downsample, BatchNorm=BatchNorm))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation, BatchNorm=BatchNorm))

        return nn.Sequential(*layers)

    def _make_pred_layer(self, block, inplanes, dilation_series, padding_series, num_classes):
        return block(inplanes, dilation_series, padding_series, num_classes)

    def cosine_similarity(self, x_relu):
        #x_relu = nn.ReLU(inplace=True)(x)
        x_norm = x_relu / (torch.norm(x_relu, dim=2, keepdim=True) + 10e-10)
        dist = x_norm.bmm(x_norm.permute(0, 2, 1))
        #去除负相关变量
        dist[dist<0]=0
        return dist

    def g_class_weight(self,prob,max,threshold=0.95):
        class_every_co=torch.zeros(self.num_classes).cuda()
        mask_threshold=(prob>=threshold)
        for i in range(self.num_classes):
            class_every_co[i]=(mask_threshold&(max==i)).sum()
        class_weight=(class_every_co/(torch.max(class_every_co)+1e-10))*threshold
        mask=(prob>=threshold)
        for i in range(self.num_classes):
            mask[(prob>=class_weight[i])&(max==i)]=True
        return mask

    def forward(self, x, class_weight=None,proportion=0.0,ssl=False, lbl=None,is_graph=True):
        if class_weight==None:
            class_weight=torch.tensor([1 for i in range(self.num_classes)]).cuda()
        _, _, h, w = x.size()
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        if self.bn_clr:
            x = self.bn_pretrain(x)

        out={}
        out['out'] = self.layer5(x, get_feat=False)
        out['feat']=self.layer6(x,get_feat=False)

        if is_graph:
            out['pred1'],out['feat1']=self.g2layer1(out['out'].clone(),out['feat'].clone(),class_weight,proportion)
            out['pred2'],out['feat2']=self.g2layer2(out['pred1'].clone(),out['feat1'].clone(),class_weight,proportion)

        #out['feat']=F.normalize(out['feat'],dim=1)
        #out['feat1'] = F.normalize(out['feat1'], dim=1)
        #out['feat2'] = F.normalize(out['feat2'], dim=1)




        # out = dict()
        # out['feat'] = x
        # x = self.layer5(x)
        
        # if not ssl:
        #     x = nn.functional.upsample(x, (h, w), mode='bilinear', align_corners=True)
        #     if lbl is not None:
        #         self.loss = self.CrossEntropy2d(x, lbl)    
        # out['out'] = x
        return out

    def get_1x_lr_params(self):

        b = []

        b.append(self.conv1)
        b.append(self.bn1)
        b.append(self.layer1)
        b.append(self.layer2)
        b.append(self.layer3)
        b.append(self.layer4)

        for i in range(len(b)):
            for j in b[i].modules():
                jj = 0
                for k in j.parameters():
                    jj += 1
                    if k.requires_grad:
                        yield k

    def get_10x_lr_params(self):

        b = []
        if self.bn_clr:
            b.append(self.bn_pretrain.parameters())    
        b.append(self.layer5.parameters())
        b.append(self.layer6.parameters())
        b.append(self.g2layer1.parameters())
        b.append(self.g2layer2.parameters())

        for j in range(len(b)):
            for i in b[j]:
                yield i

    def optim_parameters(self, args):
        return [{'params': self.get_1x_lr_params(), 'lr': args.learning_rate},
                {'params': self.get_10x_lr_params(), 'lr': 10 * args.learning_rate}]
    
    def adjust_learning_rate(self, args, optimizer, i):
        lr = args.learning_rate * ((1 - float(i) / args.num_steps) ** (args.power))
        optimizer.param_groups[0]['lr'] = lr
        if len(optimizer.param_groups) > 1:
            optimizer.param_groups[1]['lr'] = lr * 10  
            
    def CrossEntropy2d(self, predict, target, weight=None, size_average=True):
        assert not target.requires_grad
        assert predict.dim() == 4
        assert target.dim() == 3
        assert predict.size(0) == target.size(0), "{0} vs {1} ".format(predict.size(0), target.size(0))
        assert predict.size(2) == target.size(1), "{0} vs {1} ".format(predict.size(2), target.size(1))
        assert predict.size(3) == target.size(2), "{0} vs {1} ".format(predict.size(3), target.size(3))
        n, c, h, w = predict.size()
        target_mask = (target >= 0) * (target != 255)
        target = target[target_mask]
        if not target.data.dim():
            return Variable(torch.zeros(1))
        predict = predict.transpose(1, 2).transpose(2, 3).contiguous()
        predict = predict[target_mask.view(n, h, w, 1).repeat(1, 1, 1, c)].view(-1, c)
        loss = F.cross_entropy(predict, target, weight=weight, size_average=size_average)
        return loss    

def freeze_bn_func(m):
    if m.__class__.__name__.find('BatchNorm') != -1 or isinstance(m, SynchronizedBatchNorm2d)\
        or isinstance(m, nn.BatchNorm2d):
        m.weight.requires_grad = False
        m.bias.requires_grad = False

#BatchNorm=SynchronizedBatchNorm2d,nn.BatchNorm2d
def Deeplab(BatchNorm=nn.BatchNorm2d, num_classes=21, freeze_bn=False, restore_from=None, initialization=None, bn_clr=False):
    model = ResNet101(Bottleneck, [3, 4, 23, 3], num_classes, BatchNorm, bn_clr=bn_clr)
    if freeze_bn:
        model.apply(freeze_bn_func)
    # if initialization is None:
    #     pretrain_dict = model_zoo.load_url('https://download.pytorch.org/models/resnet101-5d3b4d8f.pth')
    # else:
    #     pretrain_dict = torch.load(initialization)['state_dict']
    # model_dict = {}
    # state_dict = model.state_dict()
    # for k, v in pretrain_dict.items():
    #     if k in state_dict:
    #         model_dict[k] = v
    # state_dict.update(model_dict)
    # model.load_state_dict(state_dict)
    #
    # if restore_from is not None:
    #     checkpoint = torch.load(restore_from)
    #     model.load_state_dict(checkpoint['ResNet101']["model_state"])
    #     #model.load_state_dict(checkpoint['ema'])
    
    return model
