from __future__ import print_function
import argparse
import os
import shutil
import time
import sys
import random
import math

import sklearn
import sklearn.metrics

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim as optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import matplotlib as mpl
import matplotlib.pyplot as plt

import logger
from  utils import *
from cfg import parse_cfg
from darknet import Darknet
from dataloader import *

cfgfile         = 'cfg/ussd.cfg'
weightfile      = 'backup/000202.weights'
backupdir       = 'backup'

num_train_class = 1
nsamples        = 3
num_image_class = 1313

batch_size      = 1
learning_rate   = 0.0001
momentum        = 0.1
decay           = 0.0005

width           = 416
height          = 416
channels        = 9
saturation      = 1.5
exposure        = 1.5
hue             = .1

use_cuda        = True
gpus            = '0,1,2,3'
ngpus           = 1

burn_in         = 1500

seed            = int(time.time())
eps             = 1e-5
save_interval   = 10  # epoches
dot_interval    = 70  # batches

if not os.path.exists(backupdir):
    os.mkdir(backupdir)

##############
torch.manual_seed(seed)
if use_cuda:
    os.environ['CUDA_VISIBLE_DEVICES'] = gpus
    torch.cuda.manual_seed(seed)

# create model
model = Darknet(cfgfile)
region_loss = model.loss

# model.load_weights('cfg/yolo.weights')
model.print_network()

region_loss.seen  = model.seen
processed_batches = model.seen/batch_size/nsamples

init_width        = model.width
init_height       = model.height
init_epoch        = model.seen/nsamples 

if use_cuda:
    if ngpus > 1:
        model = torch.nn.DataParallel(model).cuda()
    else:
        model = model.cuda()

total_image, total_grid = total_image_loader(num_train_class)
total_image = np.array(total_image)
total_image = Variable(torch.FloatTensor(total_image))
total_grid = np.array(total_grid)
total_grid = Variable(torch.FloatTensor(total_grid))
print('-------------------------------')
print('       Finish load data        ')

params_dict = dict(model.named_parameters())
params = []
for key, value in params_dict.items():
    if key.find('.bn') >= 0 or key.find('.bias') >= 0:
        params += [{'params': [value], 'weight_decay': 0.0}]
    else:
        params += [{'params': [value], 'weight_decay': decay*batch_size}]

optimizer = optim.SGD(model.parameters(), lr=learning_rate/batch_size, momentum=momentum, dampening=0, weight_decay=decay*batch_size)

def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = learning_rate
    lr = lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        
    return lr

def train(epoch):
    t0 = time.time()
    if ngpus > 1:
        cur_model = model.module
    else:
        cur_model = model
    
    lr = adjust_learning_rate(optimizer, epoch)
    logging('epoch %d, processed %d samples, lr %f' % (epoch, epoch * batch_size * nsamples, lr))
    model.train()
    t1 = time.time()
    avg_time = torch.zeros(9)

    num_batch = num_train_class * num_image_class / batch_size
    for batch_idx in xrange(num_batch):        
        idx = batch_idx * 3 * batch_size
        if idx + 3*batch_size > num_batch:
            break
            
        data_tmp = total_image[idx : idx + 3 * batch_size, :, :, :]
        data = data_tmp.view(batch_size, 3 * data_tmp.shape[1], data_tmp.shape[2], data_tmp.shape[3])

        target_tmp = total_grid[idx : idx + 3 * batch_size, :, :, :]
        target = target_tmp.view(batch_size, 3, target_tmp.shape[1], target_tmp.shape[2], target_tmp.shape[3])
        
        t2 = time.time()
        if use_cuda:
            data = data.cuda()
#             target= target.cuda()
        t3 = time.time()
        t4 = time.time()
        optimizer.zero_grad()
        t5 = time.time()
        output = model(data)
        if batch_idx <= 1:
            print('-----------------output------------')
            print(output)
        
        t6 = time.time()
        region_loss.seen = region_loss.seen + nsamples
        loss = region_loss(output, target)
        t7 = time.time()
        loss.backward()
        t8 = time.time()
        optimizer.step()
        t9 = time.time()
        if True and batch_idx > 1:
            avg_time[0] = avg_time[0] + (t2-t1)
            avg_time[1] = avg_time[1] + (t3-t2)
            avg_time[2] = avg_time[2] + (t4-t3)
            avg_time[3] = avg_time[3] + (t5-t4)
            avg_time[4] = avg_time[4] + (t6-t5)
            avg_time[5] = avg_time[5] + (t7-t6)
            avg_time[6] = avg_time[6] + (t8-t7)
            avg_time[7] = avg_time[7] + (t9-t8)
            avg_time[8] = avg_time[8] + (t9-t1)
            print('-------------------------------')
            print('       load data : %f' % (avg_time[0]/(batch_idx)))
            print('     cpu to cuda : %f' % (avg_time[1]/(batch_idx)))
            print('cuda to variable : %f' % (avg_time[2]/(batch_idx)))
            print('       zero_grad : %f' % (avg_time[3]/(batch_idx)))
            print(' forward feature : %f' % (avg_time[4]/(batch_idx)))
            print('    forward loss : %f' % (avg_time[5]/(batch_idx)))
            print('        backward : %f' % (avg_time[6]/(batch_idx)))
            print('            step : %f' % (avg_time[7]/(batch_idx)))
            print('           total : %f' % (avg_time[8]/(batch_idx)))
        t1 = time.time()
    print('')
    t1 = time.time()
    logging('training with %f samples/s' % (num_train_class * num_image_class/(t1-t0)))
    if (epoch+1) % save_interval == 0:
        logging('save weights to %s/%06d.weights' % (backupdir, epoch+1))
        cur_model.seen = (epoch + 1) * num_train_class * num_image_class
        cur_model.save_weights('%s/%06d.weights' % (backupdir, epoch+1))
        
def main():

    start_epoch = 0
    epoches     = 2000

    for epoch in range(start_epoch, epoches):
        train(epoch)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def metric1(output, target):
    # TODO: Ignore for now - proceed till instructed
    # MAP
    batch_size = target.size(0)
    class_types = target.size(1)
    output_cpu = output.cpu().numpy().astype('float32')
    target_cpu = target.cpu().numpy().astype('float32')
    res = 0.0
    for i in range(batch_size):
        pred_cls = output_cpu[i]
        gt_cls = target_cpu[i]
        
        ap = sklearn.metrics.average_precision_score(
            gt_cls, pred_cls, average=None)
        res += ap

    return [res/batch_size]

def metric2(output, target):
    # TODO: Ignore for now - proceed till instructed
    # Top 3 accuracy 
    res = 0.0
    for i in range(target.shape[0]):
        class_i = output[i]
        sort_idx = sorted(range(len(class_i)), key=lambda j: class_i[j])
        output_top3 = np.take(output[i], sort_idx[-3:])
        target_top3 = np.take(target[i], sort_idx[-3:])
        res= res+int((output_top3*target_top3).sum()!=0)*1.0

    return [res/target.shape[0]]

if __name__ == '__main__':
    main()
