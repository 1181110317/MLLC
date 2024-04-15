import argparse
import os
import sys
import random
import timeit
import datetime

import numpy as np
import pickle
import scipy.misc

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils import data, model_zoo
from torch.autograd import Variable
import torchvision.transforms as transform

from model.deeplabv2 import Deeplab as Res_Deeplab

from utils.loss import CrossEntropy2d
from utils.loss import CrossEntropyLoss2dPixelWiseWeighted
from utils.loss import MSELoss2d
from utils.loss import CtsEntropy2d

from utils import transformmasks
from utils import transformsgpu
from utils.helpers import colorize_mask
import utils.palette as palette

from utils.sync_batchnorm import convert_model
from utils.sync_batchnorm import DataParallelWithCallback

from data.voc_dataset import VOCDataSet

from data import get_loader, get_data_path
from data.augmentations import *
from tqdm import tqdm

import PIL
from torchvision import transforms
import json
from torch.utils import tensorboard
from evaluateSSL import evaluate

from utils.method import get_accuracy, get_meaniou

#from apex import amp

import time

from utils.feature_memory import PrototypeMemory

start = timeit.default_timer()
start_writeable = datetime.datetime.now().strftime('%m-%d_%H-%M')

def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="DeepLab-ResNet Network")
    parser.add_argument("--gpus", type=int, default=1,
                        help="choose number of gpu devices to use (default: 1)")
    parser.add_argument("-c", "--config", type=str, default='config.json',
                        help='Path to the config file (default: config.json)')
    parser.add_argument("-r", "--resume", type=str, default=None,
                        help='Path to the .pth file to resume from (default: None)')
    parser.add_argument("-n", "--name", type=str, default=None, required=True,
                        help='Name of the run (default: None)')
    parser.add_argument("--save-images", type=str, default=None,
                        help='Include to save images (default: None)')
    return parser.parse_args()


def loss_calc(pred, label):
    label = Variable(label.long()).cuda()
    if len(gpus) > 1:
        criterion = torch.nn.DataParallel(CrossEntropy2d(ignore_label=ignore_label),
                                          device_ids=gpus).cuda()  # Ignore label ??
    else:
        criterion = CrossEntropy2d(ignore_label=ignore_label).cuda()  # Ignore label ??

    return criterion(pred, label)


def loss_cts(pred, label):
    label = Variable(label.long()).cuda()
    if len(gpus) > 1:
        criterion = torch.nn.DataParallel(CtsEntropy2d(ignore_label=ignore_label),
                                          device_ids=gpus).cuda()  # Ignore label ??
    else:
        criterion = CtsEntropy2d(ignore_label=ignore_label).cuda()  # Ignore label ??

    return criterion(pred, label)


def ent_loss(pred):
    pred_logsoftmax = torch.log_softmax(pred, dim=1)
    pred_softmax = torch.softmax(pred, dim=1)
    loss = torch.sum(-pred_softmax * pred_logsoftmax) / (pred.size(0) * pred.size(2) * pred.size(3))
    return loss


def ema_loss(model, ema_model, images_remain, inputs_u_w, weak_parameters, interp, i_iter, unlabeled_loss,prototype_memory,prototype_memory1,prototype_memory2,class_weight):
    with torch.no_grad():
        out=model(inputs_u_w.detach(),class_weight)
        logits_u_w = interp(out['pred2'])
        logits_u_w, _ = weakTransform(getWeakInverseTransformParameters(weak_parameters), data=logits_u_w.detach())

        softmax_u_w = torch.softmax(logits_u_w.detach(), dim=1)
        max_probs, argmax_u_w = torch.max(softmax_u_w, dim=1)

        if mix_mask == "class":

            for image_i in range(batch_size):
                classes = torch.unique(argmax_u_w[image_i])
                classes = classes[classes != ignore_label]
                nclasses = classes.shape[0]
                classes = (classes[torch.Tensor(
                    np.random.choice(nclasses, int((nclasses - nclasses % 2) / 2), replace=False)).long()]).cuda()
                if image_i == 0:
                    MixMask = transformmasks.generate_class_mask(argmax_u_w[image_i], classes).unsqueeze(0).cuda()
                else:
                    MixMask = torch.cat(
                        (MixMask, transformmasks.generate_class_mask(argmax_u_w[image_i], classes).unsqueeze(0).cuda()))

        elif mix_mask == 'cut':
            img_size = inputs_u_w.shape[2:4]
            for image_i in range(batch_size):
                if image_i == 0:
                    MixMask = torch.from_numpy(transformmasks.generate_cutout_mask(img_size)).unsqueeze(0).cuda().float()
                else:
                    MixMask = torch.cat((MixMask, torch.from_numpy(transformmasks.generate_cutout_mask(img_size)).unsqueeze(
                        0).cuda().float()))

        elif mix_mask == "cow":
            img_size = inputs_u_w.shape[2:4]
            sigma_min = 8
            sigma_max = 32
            p_min = 0.5
            p_max = 0.5
            for image_i in range(batch_size):
                sigma = np.exp(np.random.uniform(np.log(sigma_min), np.log(sigma_max)))  # Random sigma
                p = np.random.uniform(p_min, p_max)  # Random p
                if image_i == 0:
                    MixMask = torch.from_numpy(transformmasks.generate_cow_mask(img_size, sigma, p, seed=None)).unsqueeze(
                        0).cuda().float()
                else:
                    MixMask = torch.cat((MixMask, torch.from_numpy(
                        transformmasks.generate_cow_mask(img_size, sigma, p, seed=None)).unsqueeze(0).cuda().float()))

        elif mix_mask == None:
            MixMask = torch.ones((inputs_u_w.shape)).cuda()

        strong_parameters = {"Mix": MixMask}
        #strong_parameters = {}
        if random_flip:
            strong_parameters["flip"] = random.randint(0, 1)
        else:
            strong_parameters["flip"] = 0
        if color_jitter:
            strong_parameters["ColorJitter"] = random.uniform(0, 1)
        else:
            strong_parameters["ColorJitter"] = 0
        if gaussian_blur:
            strong_parameters["GaussianBlur"] = random.uniform(0, 1)
        else:
            strong_parameters["GaussianBlur"] = 0

        inputs_u_s, _ = strongTransform(strong_parameters, data=images_remain)

    logits_u_p=model(inputs_u_s,class_weight)
    logits_u_s = interp(logits_u_p['out'])

    softmax_u_w_mixed, _ = strongTransform(strong_parameters, data=softmax_u_w)
    max_probs, pseudo_label = torch.max(softmax_u_w_mixed, dim=1)

    if pixel_weight == "threshold_uniform":
        unlabeled_weight = torch.sum(max_probs.ge(0.968).long() == 1).item() / np.size(np.array(pseudo_label.cpu()))
        pixelWiseWeight = unlabeled_weight * torch.ones(max_probs.shape).cuda()
    elif pixel_weight == "threshold":
        # pixelWiseWeight = max_probs.ge(0.968).long().cuda()
        pixelWiseWeight=torch.ones(max_probs.shape).cuda()
        pixelWiseWeight=pixelWiseWeight*torch.pow(max_probs.detach(),6)
    elif pixel_weight == 'sigmoid':
        max_iter = 10000
        pixelWiseWeight = sigmoid_ramp_up(i_iter, max_iter) * torch.ones(max_probs.shape).cuda()
    elif pixel_weight == False:
        pixelWiseWeight = torch.ones(max_probs.shape).cuda()

    if consistency_loss == 'CE':
        #labels = torch.cat([pseudo_label.unsqueeze(1).float(), max_probs.unsqueeze(1).float()], dim=1)
        L_u = consistency_weight * unlabeled_loss(logits_u_s, pseudo_label, pixelWiseWeight)
        L_u=L_u+consistency_weight*unlabeled_loss(interp(logits_u_p['pred1']),pseudo_label,pixelWiseWeight)
        L_u=L_u+consistency_weight * unlabeled_loss(interp(logits_u_p['pred2']), pseudo_label, pixelWiseWeight)

        #L_u = consistency_weight * unlabeled_loss(logits_u_s, labels,max_probs.shape[0])
    elif consistency_loss == 'MSE':
        unlabeled_weight = torch.sum(max_probs.ge(0.968).long() == 1).item() / np.size(np.array(pseudo_label.cpu()))
        # softmax_u_w_mixed = torch.cat((softmax_u_w_mixed[1].unsqueeze(0),softmax_u_w_mixed[0].unsqueeze(0)))
        L_u = consistency_weight * unlabeled_weight * unlabeled_loss(logits_u_s, softmax_u_w_mixed)

        del logits_u_w,max_probs,argmax_u_w,MixMask,softmax_u_w_mixed,softmax_u_w,strong_parameters

    prob = F.interpolate(max_probs.clone().unsqueeze(1), size=logits_u_p['feat'].shape[2:],mode='nearest').squeeze(1)
    max = F.interpolate(pseudo_label.clone().unsqueeze(1).float(), size=logits_u_p['feat'].shape[2:], mode='nearest').squeeze(1).long()

    u_prob,u_max=torch.max(torch.softmax(interp(out['out']),dim=1),dim=1)

    #L_u=L_u+contr_loss(logits_u_p['feat1'],max.float())
    L_u=L_u+pro_loss(logits_u_p['feat'],max,prototype_memory,i_iter)
    L_u = L_u + pro_loss(logits_u_p['feat1'], max, prototype_memory1,i_iter)
    L_u = L_u + pro_loss(logits_u_p['feat2'], max, prototype_memory2,i_iter)

    return L_u,torch.argmax(interp(out['out']),dim=1),torch.argmax(interp(out['pred1']),dim=1),None,out,u_prob,u_max

def contr_loss(feat,pse):
    feat=feat.permute(0,2,3,1).contiguous().view(batch_size, -1, 256)
    with torch.no_grad():
        pse = torch.where(pse == ignore_label * torch.ones_like(pse),num_classes * torch.ones_like(pse), pse)
        pse = torch.eye(num_classes + 1)[pse.long()].float().cuda().permute(0, 3, 1, 2)
        pse = pse.permute(0, 2, 3, 1).contiguous().view(batch_size, -1, num_classes + 1)
        affinity_map_label = pse.bmm(pse.permute(0, 2, 1)).detach()

    #x_relu = nn.ReLU(inplace=True)(feat)
    x_relu=feat
    x_norm = x_relu / (torch.norm(x_relu, dim=2, keepdim=True) + 10e-10)
    dist = x_norm.bmm(x_norm.permute(0, 2, 1))
    dist = dist.clamp(min=1e-10, max=1.0)


    contr = 0.1*F.binary_cross_entropy(dist, affinity_map_label)
    return contr

def pro_loss(feat,label,prototype_memory,i_iter):
    if i_iter<2000:
        return 0
    prototype = prototype_memory.get_prototype_all_torch().cuda().unsqueeze(0).repeat(batch_size,1,1)
    prototype_cirterion = nn.CrossEntropyLoss(ignore_index=ignore_label, reduction='none')
    feat=F.normalize(feat,dim=1)
    feat = feat.view(batch_size, 256, -1).permute(0, 2, 1).contiguous()

    prototype_loss = torch.mean(prototype_cirterion(10 * torch.matmul(feat, prototype.permute(0,2,1)).permute(0,2,1), label.view(batch_size,-1)))

    return 0.1*prototype_loss


def lr_poly(base_lr, iter, max_iter, power):
    return base_lr * ((1 - float(iter) / max_iter) ** (power))


def adjust_learning_rate(optimizer, i_iter):
    lr = lr_poly(learning_rate, i_iter, num_iterations, lr_power)
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) > 1:
        optimizer.param_groups[1]['lr'] = lr * 10


def sigmoid_ramp_up(iter, max_iter):
    if iter >= max_iter:
        return 1
    else:
        return np.exp(- 5 * (1 - iter / max_iter) ** 2)


def create_ema_model(model):
    ema_model = Res_Deeplab(num_classes=num_classes)

    for param in ema_model.parameters():
        param.detach_()
    mp = list(model.parameters())
    mcp = list(ema_model.parameters())
    n = len(mp)
    for i in range(0, n):
        mcp[i].data[:] = mp[i].data[:].clone()
    if len(gpus) > 1:
        if use_sync_batchnorm:
            ema_model = convert_model(ema_model)
            ema_model = DataParallelWithCallback(ema_model, device_ids=gpus)
        else:
            ema_model = torch.nn.DataParallel(ema_model, device_ids=gpus)
    return ema_model


def update_ema_variables(ema_model, model, alpha_teacher, iteration):
    # Use the "true" average until the exponential average is more correct
    alpha_teacher = min(1 - 1 / (iteration + 1), alpha_teacher)
    if len(gpus) > 1:
        for ema_param, param in zip(ema_model.module.parameters(), model.module.parameters()):
            ema_param.data[:] = alpha_teacher * ema_param[:].data[:] + (1 - alpha_teacher) * param[:].data[:]
    else:
        for ema_param, param in zip(ema_model.parameters(), model.parameters()):
            ema_param.data[:] = alpha_teacher * ema_param[:].data[:] + (1 - alpha_teacher) * param[:].data[:]
    return ema_model


def strongTransform(parameters, data=None, target=None):
    assert ((data is not None) or (target is not None))
    if 'Mix' in parameters.keys():
        data, target = transformsgpu.mix(mask=parameters["Mix"], data=data, target=target)
    data, target = transformsgpu.colorJitter(colorJitter=parameters["ColorJitter"],
                                             img_mean=torch.from_numpy(IMG_MEAN.copy()).cuda(), data=data,
                                             target=target)
    data, target = transformsgpu.gaussian_blur(blur=parameters["GaussianBlur"], data=data, target=None)
    data, target = transformsgpu.flip(flip=parameters["flip"], data=data, target=target)
    return data, target


def weakTransform(parameters, data=None, target=None):
    data, target = transformsgpu.flip(flip=parameters["flip"], data=data, target=target)
    return data, target


def getWeakInverseTransformParameters(parameters):
    return parameters


def getStrongInverseTransformParameters(parameters):
    return parameters


class DeNormalize(object):
    def __init__(self, mean):
        self.mean = mean

    def __call__(self, tensor):
        IMG_MEAN = torch.from_numpy(self.mean.copy())
        IMG_MEAN, _ = torch.broadcast_tensors(IMG_MEAN.unsqueeze(1).unsqueeze(2), tensor)
        tensor = tensor + IMG_MEAN
        tensor = (tensor / 255).float()
        tensor = torch.flip(tensor, (0,))
        return tensor


class Learning_Rate_Object(object):
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate


def save_image(image, epoch, id, palette):
    with torch.no_grad():
        if image.shape[0] == 3:
            restore_transform = transforms.Compose([
                DeNormalize(IMG_MEAN),
                transforms.ToPILImage()])

            image = restore_transform(image)
            # image = PIL.Image.fromarray(np.array(image)[:, :, ::-1])  # BGR->RGB
            image.save(os.path.join('../visualiseImages/', str(epoch) + id + '.png'))
        else:
            mask = image.numpy()
            colorized_mask = colorize_mask(mask, palette)
            colorized_mask.save(os.path.join('../visualiseImages/', str(epoch) + id + '.png'))


def _save_checkpoint(iteration, model, optimizer, config, ema_model, model_name, save_best=False, overwrite=True):
    checkpoint = {
        'iteration': iteration,
        'optimizer': optimizer.state_dict(),
        'config': config,
    }
    if len(gpus) > 1:
        checkpoint['model'] = model.module.state_dict()
        if train_unlabeled:
            checkpoint['ema_model'] = ema_model.module.state_dict()
    else:
        checkpoint['model'] = model.state_dict()
        if train_unlabeled:
            checkpoint['ema_model'] = ema_model.state_dict()

    if save_best:
        filename = os.path.join(checkpoint_dir, f'best_model.pth')
        torch.save(checkpoint, filename)
        print("Saving current best model: best_model.pth")
    else:
        filename = os.path.join(checkpoint_dir, f'checkpoint-iter{iteration}.pth')
        print(f'\nSaving a checkpoint: {filename} ...')
        torch.save(checkpoint, filename)
        if overwrite:
            try:
                os.remove(os.path.join(checkpoint_dir,
                                       f'checkpoint-iter{iteration - save_checkpoint_every - model_name}.pth'))
            except:
                pass


def _resume_checkpoint(resume_path, model, optimizer, ema_model):
    print(f'Loading checkpoint : {resume_path}')
    checkpoint = torch.load(resume_path)

    # Load last run info, the model params, the optimizer and the loggers
    iteration = checkpoint['iteration'] + 1
    print('Starting at iteration: ' + str(iteration))

    if len(gpus) > 1:
        model.module.load_state_dict(checkpoint['model'])
    else:
        model.load_state_dict(checkpoint['model'])

    optimizer.load_state_dict(checkpoint['optimizer'])

    if train_unlabeled:
        if len(gpus) > 1:
            ema_model.module.load_state_dict(checkpoint['ema_model'])
        else:
            ema_model.load_state_dict(checkpoint['ema_model'])

    return iteration, model, optimizer, ema_model

def update_class_weight(prob,max,class_weight,threshold=0.95):
    mask_threshold = (prob >= threshold)
    for i in range(num_classes):
        class_weight[i] += (mask_threshold & (max == i)).sum()

    return class_weight





def main():
    print(config)

    best_mIoU_l = 0
    best_mIoU_r = 0

    if consistency_loss == 'CE':
        if len(gpus) > 1:
            unlabeled_loss = torch.nn.DataParallel(CrossEntropyLoss2dPixelWiseWeighted(ignore_index=ignore_label),device_ids=gpus).cuda()
            #unlabeled_loss = torch.nn.DataParallel(DynamicMutualLoss(gamma1=2, gamma2=2, ignore_index=ignore_label),device_ids=gpus).cuda()
        else:
            unlabeled_loss = CrossEntropyLoss2dPixelWiseWeighted(ignore_index=ignore_label).cuda()
            #unlabeled_loss = DynamicMutualLoss(gamma1=2, gamma2=2, ignore_index=ignore_label).cuda()
    elif consistency_loss == 'MSE':
        if len(gpus) > 1:
            unlabeled_loss = torch.nn.DataParallel(MSELoss2d(), device_ids=gpus).cuda()

        else:
            unlabeled_loss = MSELoss2d().cuda()

    cudnn.enabled = True

    # create network
    model_l = Res_Deeplab(num_classes=num_classes)

    # load pretrained parameters
    if restore_from_l[:4] == 'http':
        saved_state_dict_l = model_zoo.load_url(restore_from_l)
    else:
        saved_state_dict_l = torch.load(restore_from_l)

    # Copy loaded parameters to model
    new_params = model_l.state_dict().copy()
    for name, param in new_params.items():
        if name in saved_state_dict_l and param.size() == saved_state_dict_l[name].size():
            new_params[name].copy_(saved_state_dict_l[name])
    model_l.load_state_dict(new_params)


    # Initiate ema-model
    if train_unlabeled:
        ema_model_l = create_ema_model(model_l)
        ema_model_l.train()
        ema_model_l = ema_model_l.cuda()
    else:
        ema_model_l = None

    if len(gpus) > 1:
        if use_sync_batchnorm:
            model_l = convert_model(model_l)
            model_l = DataParallelWithCallback(model_l, device_ids=gpus)

        else:
            model_l = torch.nn.DataParallel(model_l, device_ids=gpus)
    model_l.train()
    model_l.cuda()

    cudnn.benchmark = True


    if dataset == 'pascal_voc':
        data_loader = get_loader(dataset)
        data_path = get_data_path(dataset)
        train_dataset = data_loader(data_path, crop_size=input_size, scale=random_scale, mirror=random_flip)

    elif dataset == 'cityscapes':
        data_loader = get_loader('cityscapes')
        data_path = get_data_path('cityscapes')
        if random_crop:
            data_aug = RandomCrop_city(input_size)
        else:
            data_aug = None

        train_dataset = data_loader(data_path, is_transform=True, augmentations=data_aug)

    train_dataset_size = len(train_dataset)
    print('dataset size: ', train_dataset_size)

    partial_size = labeled_samples
    print('Training on number of samples:', partial_size)
    if split_id is not None:
        train_ids = pickle.load(open(split_id, 'rb'))
        print('loading train ids from {}'.format(split_id))
    else:
        np.random.seed(random_seed)
        train_ids = np.arange(train_dataset_size)
        np.random.shuffle(train_ids)

    train_sampler = data.sampler.SubsetRandomSampler(train_ids[:partial_size])
    trainloader = data.DataLoader(train_dataset,
                                  batch_size=batch_size, sampler=train_sampler, num_workers=num_workers,
                                  pin_memory=True)
    trainloader_iter = iter(trainloader)

    if train_unlabeled:
        train_remain_sampler = data.sampler.SubsetRandomSampler(train_ids[partial_size:])
        trainloader_remain = data.DataLoader(train_dataset,
                                             batch_size=batch_size, sampler=train_remain_sampler, num_workers=num_workers,
                                             pin_memory=True)
        trainloader_remain_iter = iter(trainloader_remain)

    # Optimizer for segmentation network
    learning_rate_object = Learning_Rate_Object(config['training']['learning_rate'])

    if optimizer_type == 'SGD':
        if len(gpus) > 1:
            optimizer_l = optim.SGD(model_l.module.optim_parameters(learning_rate_object),
                                    lr=learning_rate, momentum=momentum, weight_decay=weight_decay)

        else:
            optimizer_l = optim.SGD(model_l.optim_parameters(learning_rate_object),
                                    lr=learning_rate, momentum=momentum, weight_decay=weight_decay)


    #model_l, optimizer_l = amp.initialize(model_l, optimizer_l, opt_level='O1')
    #model_r, optimizer_r = amp.initialize(model_r, optimizer_r, opt_level='O1')

    #model_l = DataParallelWithCallback(model_l, device_ids=gpus)
    #model_r = DataParallelWithCallback(model_r, device_ids=gpus)

    optimizer_l.zero_grad()

    interp = nn.Upsample(size=(input_size[0], input_size[1]), mode='bilinear', align_corners=True)

    start_iteration = 0

    if args.resume:
        start_iteration, model_l, optimizer_l, ema_model_l = _resume_checkpoint(args.resume, model_l, optimizer_l,
                                                                                ema_model_l)

    accumulated_loss_l_l = []
    if train_unlabeled:
        accumulated_loss_u_l = []

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    with open(checkpoint_dir + '/config.json', 'w') as handle:
        json.dump(config, handle, indent=4, sort_keys=False)
    pickle.dump(train_ids, open(os.path.join(checkpoint_dir, 'train_split.pkl'), 'wb'))

    class_sampling = [1.0 for i in range(num_classes)]

    epochs_since_start = 0


    with_embed = False

    epoch = 0
    epochs = math.ceil(num_iterations / (labeled_samples // batch_size))

    loss_l_value_l = 0

    if train_unlabeled:
        loss_u_value_l = 0

    total = 0

    acc_l = 0
    meaniou_l = 0

    start_epoch = datetime.datetime.now()
    end_epoch = datetime.datetime.now()

    prototype_memory = PrototypeMemory(feature_size=256, n_classes=num_classes, gamma=0.99)
    prototype_memory1=PrototypeMemory(feature_size=256, n_classes=num_classes, gamma=0.99)
    prototype_memory2 = PrototypeMemory(feature_size=256, n_classes=num_classes, gamma=0.99)
    class_weight=torch.zeros(num_classes).cuda()

    for i_iter in range(start_iteration, num_iterations):
        model_l.train()

        start = datetime.datetime.now()

        optimizer_l.zero_grad()

        if lr_schedule:
            adjust_learning_rate(optimizer_l, i_iter)

        # Training loss for labeled data only
        try:
            batch = next(trainloader_iter)
            if batch[0].shape[0] != batch_size:
                batch = next(trainloader_iter)
        except:
            epochs_since_start = epochs_since_start + 1
            # print('Epochs since start: ',epochs_since_start)
            trainloader_iter = iter(trainloader)
            batch = next(trainloader_iter)

        weak_parameters = {"flip": 0}

        images, labels, _, _, n = batch
        images = images.cuda()
        labels = labels.cuda()

        images, labels = weakTransform(weak_parameters, data=images, target=labels)

        pred_l = model_l(images.detach(),class_weight=class_weight)

        L_l_l = loss_calc(interp(pred_l['out']), labels)
        L_l_l=L_l_l+loss_calc(interp(pred_l['pred1']), labels)+loss_calc(interp(pred_l['pred2']), labels)

        l_prob,l_max=torch.max(torch.softmax(interp(pred_l['out']),dim=1),dim=1)

        labels_feat = F.interpolate(labels.clone().unsqueeze(1).float(), size=pred_l['feat'].shape[2:], mode='nearest').squeeze(1).long()
        #L_l_l=L_l_l+contr_loss(pred_l['feat1'],labels_feat)
        L_l_l = L_l_l + pro_loss(pred_l['feat'], labels_feat, prototype_memory,i_iter)
        L_l_l = L_l_l + pro_loss(pred_l['feat1'], labels_feat, prototype_memory1, i_iter)
        L_l_l = L_l_l + pro_loss(pred_l['feat2'], labels_feat, prototype_memory2, i_iter)


        if train_unlabeled:
            try:
                batch_remain = next(trainloader_remain_iter)
                if batch_remain[0].shape[0] != batch_size:
                    batch_remain = next(trainloader_remain_iter)
            except:
                trainloader_remain_iter = iter(trainloader_remain)
                batch_remain = next(trainloader_remain_iter)

            images_remain, labels_remain, _, _, _ = batch_remain
            images_remain = images_remain.cuda()
            inputs_u_w, _ = weakTransform(weak_parameters, data=images_remain)

            # ema损失
            L_u_ema_l,pred_o,pred_g1,pred_g2,un_out,u_prob,u_max = ema_loss(model_l, ema_model_l, images_remain, inputs_u_w, weak_parameters, interp, i_iter,unlabeled_loss,prototype_memory,prototype_memory1,prototype_memory2,class_weight)

            loss_l = L_l_l + L_u_ema_l

        else:
            loss_l = L_l_l



        if len(gpus) > 1:
            loss_l = loss_l.mean()
            loss_l_value_l += float(L_l_l.mean().item())
        else:
            loss_l_value_l += float(L_l_l.item())

        loss_l.backward()
        # with amp.scale_loss(loss_l,optimizer_l) as scaled_loss_l:
        #     scaled_loss_l.backward()
        optimizer_l.step()



        # update Mean teacher network
        if ema_model_l is not None:
            alpha_teacher = 0.99

            ema_model_l = update_ema_variables(ema_model=ema_model_l, model=model_l, alpha_teacher=alpha_teacher,
                                               iteration=i_iter)


        argmax_l = torch.argmax(interp(pred_l['out']), dim=1)
        for j in range(batch_size):
            acc_l += get_accuracy(argmax_l[j], labels[j])
            meaniou_l += get_meaniou(argmax_l[j], labels[j], n_classes=num_classes)
            total += 1

        end = datetime.datetime.now()


        #if with_embed:
            #del L_l_contr,L_r_contr,labels_l,labels_r
            #del prob_l, prob_r, max_l, max_r, mask_l, mask_r, which_memory
            #del pred_u_l,pred_u_r,pred_rep_l,pred_rep_r
        # del loss_l, loss_r,images,images_remain,L_l_l,L_l_r,argmax_l,argmax_r
        # del labels,labels_feat
        # del pred_l,pred_r,inputs_u_w

        torch.cuda.empty_cache()


        running_time = (end - start) * (
                    (labeled_samples // batch_size) - (i_iter % (labeled_samples // batch_size)) - 1)
        running_time = str(running_time)[:7]
        p = (total / labeled_samples) * 100
        show_str = ('[%%-%ds]' % 65) % (int(65 * p / 100) * ">")

        # print(
        #     '\rEpoch [%d/%d] %d/%d %s %d%% Running time: %s [Training] Loss_l: %.4f, Loss_r: %.4f, Acc_l: %.4f, MeanIou_l: %.4f, Acc_r: %.4f, MeanIou_r: %.4f' % (
        #         epoch + 1, epochs, total, labeled_samples, show_str, p, running_time,
        #         loss_l_value_l / (total // batch_size),
        #         0, acc_l / total, meaniou_l / total, 0,
        #         0), end='')

        labels_remain=labels_remain.cuda()
        size=labels_remain.size(0)*labels_remain.size(1)*labels_remain.size(2)

        pred_o_acc=(labels_remain==pred_o).sum()/size
        pred_g1_acc=(labels_remain==pred_g1).sum()/size
        #pred_g2_acc=(labels_remain==pred_g2).sum()/size
        pred_g2_acc=(labels_remain==pred_g2).sum()/size

        print('\rtraining..........   iter = %d/%d    pred_acc1: %s   pred_acc2: %s   pred_acc3: %s' % (
            i_iter, num_iterations, pred_o_acc.item(), pred_g1_acc.item(),
            pred_g2_acc.item()), end='')
        unlabeled_feat_label=torch.argmax(un_out['pred2'],dim=1)
        prototype_memory.update_prototype(torch.cat([pred_l['feat'],un_out['feat']]),torch.cat([labels_feat,unlabeled_feat_label]))
        prototype_memory.update_prototype(torch.cat([pred_l['feat1'], un_out['feat1']]),
                                          torch.cat([labels_feat, unlabeled_feat_label]))
        prototype_memory.update_prototype(torch.cat([pred_l['feat2'], un_out['feat2']]),
                                          torch.cat([labels_feat, unlabeled_feat_label]))

        class_weight=update_class_weight(l_prob,l_max,class_weight)
        class_weight=update_class_weight(u_prob,u_max,class_weight)
        if i_iter != 0 and total >= labeled_samples:
            end_epoch = datetime.datetime.now()
            running_time = end_epoch - start_epoch
            running_time = str(running_time)[:7]
            p = (total / labeled_samples) * 100
            show_str = ('[%%-%ds]' % 65) % (int(65 * p / 100) * ">")

            # print(
            #     '\rEpoch [%d/%d] %d/%d %s %d%% Running time: %s [Training] Loss_l: %.4f, Loss_r: %.4f, Acc_l: %.4f, MeanIou_l: %.4f, Acc_r: %.4f, MeanIou_r: %.4f' % (
            #         epoch + 1, epochs, total, labeled_samples, show_str, p, running_time,
            #         loss_l_value_l / (total // batch_size),
            #         0, acc_l / total, meaniou_l / total, 0,
            #         0), end='')
            print()
            epoch += 1
            loss_l_value_l = 0
            loss_l_value_r = 0
            if train_unlabeled:
                loss_u_value_l = 0
                loss_u_value_r = 0
            total = 0
            acc_l = 0
            acc_r = 0
            meaniou_l = 0
            meaniou_r = 0

            if epoch % 10 == 0 and i_iter != 0:
                model_l.eval()
                mIoU_l, eval_loss_l = evaluate(model_l, dataset, model_name='l', ignore_label=ignore_label,
                                               input_size=(512, 1024), save_dir=checkpoint_dir)

                model_l.train()

                if mIoU_l > best_mIoU_l and save_best_model:
                    best_mIoU_l = mIoU_l
                    _save_checkpoint(i_iter, model_l, optimizer_l, config, ema_model_l, save_best=True, model_name='l')

                if use_tensorboard:
                    tensorboard_writer.add_scalar('Validation/mIoU_l', mIoU_l, i_iter)
                    tensorboard_writer.add_scalar('Validation/Loss_l', eval_loss_l, i_iter)

            start_epoch = datetime.datetime.now()
            end_epoch = datetime.datetime.now()

        if i_iter % save_checkpoint_every == 0 and i_iter != 0:
            _save_checkpoint(i_iter, model_l, optimizer_l, config, ema_model_l, 'l')

        if use_tensorboard:
            if 'tensorboard_writer' not in locals():
                tensorboard_writer = tensorboard.SummaryWriter(log_dir, flush_secs=30)

            accumulated_loss_l_l.append(loss_l_value_l)
            if train_unlabeled:
                accumulated_loss_u_l.append(loss_u_value_l)
            if i_iter % log_per_iter == 0 and i_iter != 0:

                tensorboard_writer.add_scalar('Training/Supervised loss_l', np.mean(accumulated_loss_l_l), i_iter)
                accumulated_loss_l_l = []

                if train_unlabeled:
                    tensorboard_writer.add_scalar('Training/Unsupervised loss_l', np.mean(accumulated_loss_u_l), i_iter)
                    accumulated_loss_u_l = []
                    accumulated_loss_u_l = []

        if i_iter == num_iterations - 1:
            model_l.eval()
            mIoU_l, eval_loss_l = evaluate(model_l, dataset, model_name='l', ignore_label=ignore_label,
                                           input_size=(512, 1024), save_dir=checkpoint_dir)

            model_l.train()


            if mIoU_l > best_mIoU_l and save_best_model:
                best_mIoU_l = mIoU_l
                _save_checkpoint(i_iter, model_l, optimizer_l, config, ema_model_l, save_best=True, model_name='l')


            if use_tensorboard:
                tensorboard_writer.add_scalar('Validation/mIoU_l', mIoU_l, i_iter)
                tensorboard_writer.add_scalar('Validation/Loss_l', eval_loss_l, i_iter)


        inputs_u_s_l=torch.randn(2,3,256,512)
        inputs_u_s_r=torch.randn(2,3,256,512)
        logits_u_s_l=torch.randn(2,19,256,512)
        logits_u_s_r=torch.randn(2,19,256,512)
        if save_unlabeled_images and train_unlabeled and i_iter % save_checkpoint_every == 0:
            # Saves two mixed images and the corresponding prediction
            save_image(inputs_u_s_l[0].cpu(), i_iter, 'input1_l', palette.CityScpates_palette)
            save_image(inputs_u_s_l[1].cpu(), i_iter, 'input2_l', palette.CityScpates_palette)
            _, pred_u_s = torch.max(logits_u_s_l, dim=1)
            save_image(pred_u_s[0].cpu(), i_iter, 'pred1_l', palette.CityScpates_palette)
            save_image(pred_u_s[1].cpu(), i_iter, 'pred2_l', palette.CityScpates_palette)

            save_image(inputs_u_s_r[0].cpu(), i_iter, 'input1_r', palette.CityScpates_palette)
            save_image(inputs_u_s_r[1].cpu(), i_iter, 'input2_r', palette.CityScpates_palette)
            _, pred_u_s = torch.max(logits_u_s_r, dim=1)
            save_image(pred_u_s[0].cpu(), i_iter, 'pred1_r', palette.CityScpates_palette)
            save_image(pred_u_s[1].cpu(), i_iter, 'pred2_r', palette.CityScpates_palette)

    _save_checkpoint(num_iterations, model_l, optimizer_l, config, ema_model_l, model_name='l')

    model_l.eval()
    mIoU_l, val_loss_l = evaluate(model_l, dataset, model_name='l', ignore_label=ignore_label, input_size=(512, 1024),
                                  save_dir=checkpoint_dir)
    model_l.train()
    if mIoU_l > best_mIoU_l and save_best_model:
        best_mIoU_l = mIoU_l
        _save_checkpoint(i_iter, model_l, optimizer_l, config, ema_model_l, save_best=True, model_name='l')

    if use_tensorboard:
        tensorboard_writer.add_scalar('Validation/mIoU_l', mIoU_l, i_iter)
        tensorboard_writer.add_scalar('Validation/Loss_l', val_loss_l, i_iter)

    end = timeit.default_timer()
    print('Total time: ' + str(end - start) + ' seconds')


if __name__ == '__main__':

    print('---------------------------------Starting---------------------------------')

    args = get_arguments()

    if args.resume:
        config = torch.load(args.resume)['config']
    else:
        config = json.load(open(args.config))

    model = config['model']
    dataset = config['dataset']

    if dataset == 'cityscapes':
        IMG_MEAN = np.flip(np.array([73.15835921, 82.90891754, 72.39239876]))
        num_classes = 19
        if config['training']['data']['split_id_list'] == 0:
            split_id = './splits/city/split_0.pkl'
        elif config['training']['data']['split_id_list'] == 1:
            split_id = './splits/city/split_1.pkl'
        elif config['training']['data']['split_id_list'] == 2:
            split_id = './splits/city/split_2.pkl'
        else:
            split_id = None

    elif dataset == 'pascal_voc':
        IMG_MEAN = np.flip(np.array([104.00698793, 116.66876762, 122.67891434]))
        num_classes = 21
        data_dir = './data/voc_dataset/'
        data_list_path = './data/voc_list/train_aug.txt'
        if config['training']['data']['split_id_list'] == 1:
            split_id = './splits/voc/split_1.pkl'
        else:
            split_id = None

    if config['pretrained'] == 'coco':
        restore_from_l = 'http://vllab1.ucmerced.edu/~whung/adv-semi-seg/resnet101COCO-41f33a49.pth'
        restore_from_r = 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth'

    batch_size = config['training']['batch_size']
    num_iterations = config['training']['num_iterations']

    input_size_string = config['training']['data']['input_size']
    h, w = map(int, input_size_string.split(','))
    input_size = (h, w)

    ignore_label = config['ignore_label']  # 255 for PASCAL-VOC / 250 for Cityscapes

    learning_rate = config['training']['learning_rate']

    optimizer_type = config['training']['optimizer']
    lr_schedule = config['training']['lr_schedule']
    lr_power = config['training']['lr_schedule_power']
    weight_decay = config['training']['weight_decay']
    momentum = config['training']['momentum']
    num_workers = config['training']['num_workers']
    use_sync_batchnorm = config['training']['use_sync_batchnorm']
    random_seed = config['seed']

    labeled_samples = config['training']['data']['labeled_samples']

    # unlabeled CONFIGURATIONS
    train_unlabeled = config['training']['unlabeled']['train_unlabeled']
    mix_mask = config['training']['unlabeled']['mix_mask']
    if mix_mask == "None":
        mix_mask = None
    pixel_weight = config['training']['unlabeled']['pixel_weight']
    consistency_loss = config['training']['unlabeled']['consistency_loss']
    consistency_weight = config['training']['unlabeled']['consistency_weight']
    random_flip = config['training']['unlabeled']['flip']
    color_jitter = config['training']['unlabeled']['color_jitter']
    gaussian_blur = config['training']['unlabeled']['blur']
    #warmup_iters = config["training"]["warmup_iters"]

    random_scale = config['training']['data']['scale']
    random_crop = config['training']['data']['crop']

    save_checkpoint_every = config['utils']['save_checkpoint_every']
    if args.resume:
        checkpoint_dir = os.path.join(*args.resume.split('/')[:-1]) + '_resume-' + start_writeable
    else:
        checkpoint_dir = os.path.join(config['utils']['checkpoint_dir'], start_writeable + '-' + args.name)
    log_dir = checkpoint_dir

    val_per_iter = config['utils']['val_per_iter']
    use_tensorboard = config['utils']['tensorboard']
    log_per_iter = config['utils']['log_per_iter']

    save_best_model = config['utils']['save_best_model']
    if args.save_images:
        print('Saving unlabeled images')
        save_unlabeled_images = True
    else:
        save_unlabeled_images = False

    gpus = (0, 1, 2, 3)[:args.gpus]

    main()
