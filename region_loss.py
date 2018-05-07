import time
import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from utils import *
import numpy as np

# gx0, gy0: ground truth point, px0, py0:predicted point 
def compute_conf(gx,gy,px,py):
    DTx = np.sqrt((gx-px)**2 + (gy-py)**2)
    if DTx < 30:
        return np.exp(2 - float(DTx)/15.0)
    else:
        return 0

def build_targets(pred_boxes, target, num_anchors, num_classes, nH, nW, noobject_scale, object_scale, seen):
    nB = target.size(0)
    nA = num_anchors
    nC = num_classes
    # anchor_step = len(anchors)/num_anchors
    conf_mask  = torch.ones(nB, nA, nH, nW) * noobject_scale
    coor_mask  = torch.zeros(nB, nA, nH, nW)
    cls_mask   = torch.zeros(nB, nA, nH, nW)
    tcls       = torch.zeros(nB, nA, nH, nW) 
    tconf      = torch.zeros(nB, nA, nH, nW)
    tx0        = torch.zeros(nB, nA, nH, nW) 
    ty0        = torch.zeros(nB, nA, nH, nW) 
    tx1        = torch.zeros(nB, nA, nH, nW) 
    ty1        = torch.zeros(nB, nA, nH, nW)
    tx2        = torch.zeros(nB, nA, nH, nW) 
    ty2        = torch.zeros(nB, nA, nH, nW) 
    tx3        = torch.zeros(nB, nA, nH, nW) 
    ty3        = torch.zeros(nB, nA, nH, nW) 
    tx4        = torch.zeros(nB, nA, nH, nW) 
    ty4        = torch.zeros(nB, nA, nH, nW) 
    tx5        = torch.zeros(nB, nA, nH, nW) 
    ty5        = torch.zeros(nB, nA, nH, nW) 
    tx6        = torch.zeros(nB, nA, nH, nW) 
    ty6        = torch.zeros(nB, nA, nH, nW) 
    tx7        = torch.zeros(nB, nA, nH, nW) 
    ty7        = torch.zeros(nB, nA, nH, nW)
    tx8        = torch.zeros(nB, nA, nH, nW) 
    ty8        = torch.zeros(nB, nA, nH, nW)    

    nAnchors = nA*nH*nW
    nPixels  = nH*nW
    
#     if seen <= 1000:
#         cur_object_scale = 0
#     else:
#         cur_object_scale = object_scale
    
    cur_object_scale = 0

    for b in xrange(nB):
        for a in xrange(nA):
            for h in xrange(nH):
                for w in xrange(nW):
                    if target[b][a][h][w][0] != 0:
                        tcls[b][a][h][w]      = target[b][a][h][w][0] 
                        cls_mask[b][a][h][w]  = 1
                        coor_mask[b][a][h][w] = 1
#                         conf_mask[b][a][h][w] = object_scale

                    gx0 = target[b][a][h][w][1] 
                    gy0 = target[b][a][h][w][2] 
                    gx1 = target[b][a][h][w][3] 
                    gy1 = target[b][a][h][w][4] 
                    gx2 = target[b][a][h][w][5] 
                    gy2 = target[b][a][h][w][6] 
                    gx3 = target[b][a][h][w][7] 
                    gy3 = target[b][a][h][w][8] 
                    gx4 = target[b][a][h][w][9] 
                    gy4 = target[b][a][h][w][10]
                    gx5 = target[b][a][h][w][11]
                    gy5 = target[b][a][h][w][12]
                    gx6 = target[b][a][h][w][13]
                    gy6 = target[b][a][h][w][14]
                    gx7 = target[b][a][h][w][15]
                    gy7 = target[b][a][h][w][16]
                    gx8 = target[b][a][h][w][17]
                    gy8 = target[b][a][h][w][18]

                    tx0[b][a][h][w] =  gx0 - w 
                    ty0[b][a][h][w] =  gy0 - h 
                    tx1[b][a][h][w] =  gx1 - w 
                    ty1[b][a][h][w] =  gy1 - h
                    tx2[b][a][h][w] =  gx2 - w 
                    ty2[b][a][h][w] =  gy2 - h 
                    tx3[b][a][h][w] =  gx3 - w 
                    ty3[b][a][h][w] =  gy3 - h 
                    tx4[b][a][h][w] =  gx4 - w 
                    ty4[b][a][h][w] =  gy4 - h 
                    tx5[b][a][h][w] =  gx5 - w 
                    ty5[b][a][h][w] =  gy5 - h 
                    tx6[b][a][h][w] =  gx6 - w 
                    ty6[b][a][h][w] =  gy6 - h 
                    tx7[b][a][h][w] =  gx7 - w 
                    ty7[b][a][h][w] =  gy7 - h
                    tx8[b][a][h][w] =  gx8 - w 
                    ty8[b][a][h][w] =  gy8 - h  

                    pred_box = pred_boxes[b*nAnchors+a*nPixels+h*nW+w]

                    conf = 0
                    conf += compute_conf(gx0 - w, gy0 - h, pred_box[0], pred_box[1])
                    conf += compute_conf(gx1 - w, gy1 - h, pred_box[2], pred_box[3])
                    conf += compute_conf(gx2 - w, gy2 - h, pred_box[4], pred_box[5])
                    conf += compute_conf(gx3 - w, gy3 - h, pred_box[6], pred_box[7])
                    conf += compute_conf(gx4 - w, gy4 - h, pred_box[8], pred_box[9])
                    conf += compute_conf(gx5 - w, gy5 - h, pred_box[10], pred_box[11])
                    conf += compute_conf(gx6 - w, gy6 - h, pred_box[12], pred_box[13])
                    conf += compute_conf(gx7 - w, gy7 - h, pred_box[14], pred_box[15])
                    conf += compute_conf(gx8 - w, gy8 - h, pred_box[16], pred_box[17])
                    tconf[b][a][h][w] = conf/9.0

    return tcls,tconf,cls_mask,conf_mask,coor_mask,tx0,ty0,tx1,ty1,tx2,ty2,tx3,ty3,tx4,ty4,tx5,ty5,tx6,ty6,tx7,ty7,tx8,ty8

class RegionLoss(nn.Module):
    def __init__(self, num_classes=0, anchors=[], num_anchors=1):
        super(RegionLoss, self).__init__()
        self.num_classes = num_classes
        self.anchors = anchors
        self.num_anchors = num_anchors
        self.coord_scale = 1
        self.noobject_scale = 0.1
        self.object_scale = 5
        self.class_scale = 1
        self.seen = 0

    def forward(self, output, target):
        #output : BxAs*(18+1+num_classes)*H*W
        t0 = time.time()
        nB = output.data.size(0)
        nA = self.num_anchors
        nC = self.num_classes
        nH = output.data.size(2)
        nW = output.data.size(3)

        output   = output.view(nB, nA, (19+nC), nH, nW)

        # get class data
        cls  = output.index_select(2, Variable(torch.linspace(0,nC-1,nC).long().cuda()))
        cls  = cls.view(nB*nA, nC, nH*nW).transpose(1,2).contiguous().view(nB*nA*nH*nW, nC)

        # get conf data
        conf = output.index_select(2, Variable(torch.cuda.LongTensor([nC]))).view(nB, nA, nH, nW)

        # get centroid coor 
        x0   = F.sigmoid(output.index_select(2, Variable(torch.cuda.LongTensor([nC+1]))).view(nB, nA, nH, nW))
        y0   = F.sigmoid(output.index_select(2, Variable(torch.cuda.LongTensor([nC+2]))).view(nB, nA, nH, nW))

        # get corner coor 
        x1   = output.index_select(2, Variable(torch.cuda.LongTensor([nC+3]))).view(nB, nA, nH, nW)
        y1   = output.index_select(2, Variable(torch.cuda.LongTensor([nC+4]))).view(nB, nA, nH, nW)
        x2   = output.index_select(2, Variable(torch.cuda.LongTensor([nC+5]))).view(nB, nA, nH, nW)
        y2   = output.index_select(2, Variable(torch.cuda.LongTensor([nC+6]))).view(nB, nA, nH, nW)
        x3   = output.index_select(2, Variable(torch.cuda.LongTensor([nC+7]))).view(nB, nA, nH, nW)
        y3   = output.index_select(2, Variable(torch.cuda.LongTensor([nC+8]))).view(nB, nA, nH, nW)
        x4   = output.index_select(2, Variable(torch.cuda.LongTensor([nC+9]))).view(nB, nA, nH, nW)
        y4   = output.index_select(2, Variable(torch.cuda.LongTensor([nC+10]))).view(nB, nA, nH, nW)
        x5   = output.index_select(2, Variable(torch.cuda.LongTensor([nC+11]))).view(nB, nA, nH, nW)
        y5   = output.index_select(2, Variable(torch.cuda.LongTensor([nC+12]))).view(nB, nA, nH, nW)
        x6   = output.index_select(2, Variable(torch.cuda.LongTensor([nC+13]))).view(nB, nA, nH, nW)
        y6   = output.index_select(2, Variable(torch.cuda.LongTensor([nC+14]))).view(nB, nA, nH, nW)
        x7   = output.index_select(2, Variable(torch.cuda.LongTensor([nC+15]))).view(nB, nA, nH, nW)
        y7   = output.index_select(2, Variable(torch.cuda.LongTensor([nC+16]))).view(nB, nA, nH, nW)
        x8   = output.index_select(2, Variable(torch.cuda.LongTensor([nC+17]))).view(nB, nA, nH, nW)
        y8   = output.index_select(2, Variable(torch.cuda.LongTensor([nC+18]))).view(nB, nA, nH, nW)
        t1 = time.time()

        pred_boxes = torch.cuda.FloatTensor(18, nB*nA*nH*nW)
        grid_x = torch.linspace(0, nW-1, nW).repeat(nH,1).repeat(nB*nA, 1, 1).view(nB*nA*nH*nW).cuda()
        grid_y = torch.linspace(0, nH-1, nH).repeat(nW,1).t().repeat(nB*nA, 1, 1).view(nB*nA*nH*nW).cuda()

        pred_boxes[0]  = x0.data + grid_x
        pred_boxes[1]  = y0.data + grid_y
        pred_boxes[2]  = x1.data + grid_x
        pred_boxes[3]  = y1.data + grid_y
        pred_boxes[4]  = x2.data + grid_x
        pred_boxes[5]  = y2.data + grid_y
        pred_boxes[6]  = x3.data + grid_x
        pred_boxes[7]  = y3.data + grid_y
        pred_boxes[8]  = x4.data + grid_x
        pred_boxes[9]  = y4.data + grid_y
        pred_boxes[10] = x5.data + grid_x
        pred_boxes[11] = y5.data + grid_y
        pred_boxes[12] = x6.data + grid_x
        pred_boxes[13] = y6.data + grid_y
        pred_boxes[14] = x7.data + grid_x
        pred_boxes[15] = y7.data + grid_y
        pred_boxes[16] = x8.data + grid_x
        pred_boxes[17] = y8.data + grid_y

        pred_boxes = convert2cpu(pred_boxes.transpose(0,1).contiguous().view(-1,18))
        t2 = time.time()

        tcls,tconf,cls_mask,conf_mask,coor_mask,tx0,ty0,tx1,ty1,tx2,ty2,tx3,ty3,tx4,ty4,tx5,ty5,tx6,ty6,tx7,ty7,tx8,ty8 = \
                        build_targets(pred_boxes, target.data, nA, nC, nH, nW, self.noobject_scale, self.object_scale, self.seen)
        
        cls_mask = (cls_mask == 1)
        nProposals = int((conf > 0.25).sum().data[0])

        tx0    = Variable(tx0.cuda())
        ty0    = Variable(ty0.cuda())
        tx1    = Variable(tx1.cuda())
        ty1    = Variable(ty1.cuda())
        tx2    = Variable(tx2.cuda())
        ty2    = Variable(ty2.cuda())
        tx3    = Variable(tx3.cuda())
        ty3    = Variable(ty3.cuda())
        tx4    = Variable(tx4.cuda())
        ty4    = Variable(ty4.cuda())
        tx5    = Variable(tx5.cuda())
        ty5    = Variable(ty5.cuda())
        tx6    = Variable(tx6.cuda())
        ty6    = Variable(ty6.cuda())
        tx7    = Variable(tx7.cuda())
        ty7    = Variable(ty7.cuda())
        tx8    = Variable(tx8.cuda())
        ty8    = Variable(ty8.cuda())

        tconf = Variable(tconf.cuda())
        tcls  = Variable(tcls.view(-1)[cls_mask].long().cuda())

        conf_mask  = Variable(conf_mask.cuda())
        coor_mask  = Variable(coor_mask.cuda())
        cls_mask   = Variable(cls_mask.view(-1, 1).repeat(1,nC).cuda())
        cls        = cls[cls_mask].view(-1, nC)  
        t3 = time.time()
        
#         print('----------------x0------------------')
#         print(x0)
#         print('----------------tx0-----------------')
#         print(tx0)
#         print('----------------x8------------------')
#         print(x8)
#         print('----------------tx8-----------------')
#         print(tx8)
#         print('----------------coor----------------')
#         print(coor_mask)

        loss_x0 = self.coord_scale * nn.MSELoss(size_average=False)(x0*coor_mask, tx0*coor_mask)/2.0
        loss_y0 = self.coord_scale * nn.MSELoss(size_average=False)(y0*coor_mask, ty0*coor_mask)/2.0
        loss_x1 = self.coord_scale * nn.MSELoss(size_average=False)(x1*coor_mask, tx1*coor_mask)/2.0
        loss_y1 = self.coord_scale * nn.MSELoss(size_average=False)(y1*coor_mask, ty1*coor_mask)/2.0
        loss_x2 = self.coord_scale * nn.MSELoss(size_average=False)(x2*coor_mask, tx2*coor_mask)/2.0
        loss_y2 = self.coord_scale * nn.MSELoss(size_average=False)(y2*coor_mask, ty2*coor_mask)/2.0
        loss_x3 = self.coord_scale * nn.MSELoss(size_average=False)(x3*coor_mask, tx3*coor_mask)/2.0
        loss_y3 = self.coord_scale * nn.MSELoss(size_average=False)(y3*coor_mask, ty3*coor_mask)/2.0
        loss_x4 = self.coord_scale * nn.MSELoss(size_average=False)(x4*coor_mask, tx4*coor_mask)/2.0
        loss_y4 = self.coord_scale * nn.MSELoss(size_average=False)(y4*coor_mask, ty4*coor_mask)/2.0
        loss_x5 = self.coord_scale * nn.MSELoss(size_average=False)(x5*coor_mask, tx5*coor_mask)/2.0
        loss_y5 = self.coord_scale * nn.MSELoss(size_average=False)(y5*coor_mask, ty5*coor_mask)/2.0
        loss_x6 = self.coord_scale * nn.MSELoss(size_average=False)(x6*coor_mask, tx6*coor_mask)/2.0
        loss_y6 = self.coord_scale * nn.MSELoss(size_average=False)(y6*coor_mask, ty6*coor_mask)/2.0
        loss_x7 = self.coord_scale * nn.MSELoss(size_average=False)(x7*coor_mask, tx7*coor_mask)/2.0
        loss_y7 = self.coord_scale * nn.MSELoss(size_average=False)(y7*coor_mask, ty7*coor_mask)/2.0
        loss_x8 = self.coord_scale * nn.MSELoss(size_average=False)(x8*coor_mask, tx8*coor_mask)/2.0
        loss_y8 = self.coord_scale * nn.MSELoss(size_average=False)(y8*coor_mask, ty8*coor_mask)/2.0

        loss_conf = nn.MSELoss(size_average=False)(conf*conf_mask, tconf*conf_mask)/2.0
#         loss_cls = self.class_scale * nn.CrossEntropyLoss(size_average=False)(cls, tcls)

        loss = loss_x0 + loss_y0 + loss_x1 + loss_y1 + loss_x2 + loss_y2 + loss_x3 + loss_y3 + \
                loss_x4 + loss_y4 + loss_x5 + loss_y5 + loss_x6 + loss_y6 + loss_x7 +loss_y7 + \
                loss_x8 + loss_y8 + loss_conf
        t4 = time.time()
        if True:
            print('-----------------------------------')
            print('        activation : %f' % (t1 - t0))
            print(' create pred_boxes : %f' % (t2 - t1))
            print('     build targets : %f' % (t3 - t2))
            print('       create loss : %f' % (t4 - t3))
            print('             total : %f' % (t4 - t0))
        print('%d: loss_x0: %f, y0 %f, x1 %f, y1 %f, x2 %f, y2 %f, x3 %f, y3 %f, x4 %f, y4 %f, x5 %f, y5 %f, x6 %f, y6 %f, x7 %f, y7 %f, x8 %f, y8 %f, conf %f, total %f' % (self.seen, loss_x0.data[0], loss_y0.data[0], loss_x1.data[0], loss_y1.data[0], loss_x2.data[0], loss_y2.data[0], loss_x3.data[0], loss_y3.data[0], loss_x4.data[0], loss_y4.data[0], loss_x5.data[0], loss_y5.data[0], loss_x6.data[0], loss_y6.data[0], loss_x7.data[0], loss_y7.data[0], loss_x8.data[0], loss_y8.data[0], loss_conf.data[0], loss.data[0]))
        return loss
