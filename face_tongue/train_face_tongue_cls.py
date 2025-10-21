# -*- coding: utf-8 -*-
from __future__ import print_function, division
import argparse
from collections import OrderedDict
import sys
import os

from mobilenet.mobilenetv4 import MobileNetV4
current_dir = os.path.dirname(os.path.abspath(__file__))
current_dir_p=os.path.dirname(current_dir)
os.chdir(current_dir)
print('current_dir',current_dir)
paths = [current_dir,current_dir_p]
 
paths.append(os.path.join(current_dir, 'src'))
 
for path in paths:
    sys.path.insert(0, path)
    os.environ['PYTHONPATH'] = (os.environ.get('PYTHONPATH', '') + ':' + path).strip(':')
import configparser

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.optim import AdamW
from torch.utils.data import dataloader
import torch.nn.functional as torch_F
from face_tongue.utils.AdaBelief import AdaBelief
import time

from face_tongue.utils.market1501 import Market1501
from log import logger
from face_tongue.utils.model import ft_net

def train_model(model,  optimizer, scheduler, num_epochs=25):
    since = time.time()
    criterion = nn.CrossEntropyLoss()
    best_model_wts = model.state_dict()
    best_acc = 0.5
    sbd_config = configparser.ConfigParser()
    # sbd_config.read('./train_config.ini')
    # sbd_config.set('train', "lr", str(optimizer.param_groups[0]['lr']))
    for epoch in range(num_epochs):
        # Each epoch has a training and validation phase
        need_val=False
        m_dataset.set_epoch(epoch)

        if epoch %m_t_max==0 and epoch>0:
            scheduler.last_epoch=0
        for phase in ['train', 'val']:
            if phase == 'train':
                if epoch>0:
                    scheduler.step()
                model.train(True)  # Set model to training mode
                loader = train_loader
            else:
                if not need_val:
                    continue
                loader=test_loader
                model.train(False)  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            map_score=[[0,0.01] for i in range(opt.num_classes)]

            batch_len=len(loader)

            for step, data in enumerate(loader):
                # get the inputs
                inputs, labels,paths = data
                if use_gpu:
                    inputs = Variable(inputs.cuda())
                    labels = Variable(labels.cuda())
                else:
                    inputs, labels = Variable(inputs), Variable(labels)
                optimizer.zero_grad()
                out1 = model(inputs)

                loss = criterion(out1, labels)
                # loss =loss+ criterion(out2, labels2)
                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()
                # statistics
                lr = scheduler.get_lr()[0]
                running_loss += loss.cpu().item()
                outputs = torch_F.softmax(out1, dim=1)
                scores, preds = torch.max(outputs.data, 1)
                threshold=0.4
                correct = ((preds == labels) & (scores > threshold)).sum().item()  # 计算满足条件的正确预测数
                total = labels.size(0)  # 总样本数
                current_correct = correct / total  # 计算准确率
                running_corrects += current_correct
                if step%2==0:
                    print("{} Epoch {}/{} batch {}/{} Loss:{:.4f} acc_cur:{:.4f} avacc:{:.4f} lr:{:.2e}".format(phase,epoch,num_epochs - 1,step,batch_len,loss.item(),current_correct,running_corrects/(step+1),lr))


            if running_corrects/(step+1)>0.7:
                need_val=True
            else:
                need_val = False
            # deep copy the model
            if phase == 'val':
                if running_corrects/(step+1)<best_acc:
                    continue

                scores_str = "{:.4f}_".format(float(running_corrects/(step+1)))
                for socre in map_score:
                    scores_str+= "{:.4f}_".format(float(socre[0])/socre[1])
                best_acc=running_corrects/(step+1)
                last_model_wts = model.state_dict()
                save_network(model, epoch,scores_str[:-1])
                #draw_curve(epoch)

    time_elapsed = time.time() - since
    logger.info(f"training complete in {time_elapsed // 60}m {time_elapsed % 60}s")
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    # print('Best val Acc: {:4f}'.format(best_acc))

    return model

x_epoch = []

def save_network(network, epoch_label,acc_str):
    from datetime import datetime
    date_str = datetime.now().strftime("%m%d")
    print(date_str)  # 输出示例：0615
    dir_name = f'./models_tiaosheng_{date_str}/'+name
    os.makedirs(dir_name, exist_ok=True)
    save_filename = '%s_%s.pth' % (acc_str,epoch_label)
    save_path = os.path.join(dir_name, save_filename)
    print("模型保存地址，",save_path)
    torch.save(network.cpu().state_dict(), save_path)
    if torch.cuda.is_available:
        network.cuda()


if __name__ == "__main__":
# criterion = FocalLoss()
# criterion = LabelSmoothing(smoothing= 0.05)
    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('--gpu_ids', default='0', type=str, help='gpu_ids: e.g. 0  0,1,2  0,2')
    parser.add_argument('--name', default='mobilenet_m', type=str, help='FaceBox_s pelee')
    parser.add_argument('--data_dir', default=r"face_tongue_imgs", type=str, help='training dir path')
    parser.add_argument('--train_all', action='store_true', help='use all training data')
    parser.add_argument('--color_jitter', action='store_true', help='use color jitter in training')
    parser.add_argument('--batchsize', default=128, type=int, help='batchsize')
    parser.add_argument('--erasing_p', default=0, type=float, help='Random Erasing probability, in [0,1]')
    parser.add_argument('--use_dense', action='store_true', help='use densenet')
    parser.add_argument('--num_classes', type=int, default=4, help='')
    opt = parser.parse_args()

    data_dir = opt.data_dir
    name = opt.name

    # gpus = usegpu(gpu_count=1)

    m_dataset=Market1501(data_dir, None, "train")
    train_loader = dataloader.DataLoader(m_dataset,shuffle=True,batch_size=opt.batchsize,num_workers=0)

    m_dataset=Market1501(data_dir, None, "test")
    test_loader = dataloader.DataLoader(m_dataset, batch_size=opt.batchsize,num_workers=0)

    use_gpu = torch.cuda.is_available()

    weights=""
    if name=="pelee_2_6_a":
        model = ft_net(opt.num_classes)
        # model=PeleeNet_cls(num_classes=opt.num_classes,drop_rate=0.3

    elif name=="mobilenet_m":
        model = MobileNetV4('MobileNetV4ConvMedium', cls_num=opt.num_classes)
        pth_path=r"G:\txd\RopeCount\tiaosheng_new\model\MobileNetV4ConvMedium.pth"
        state_dict = torch.load(pth_path)
        try:
            model.load_state_dict(state_dict)
        except Exception as e:
            print("load not good ,start strict=False ")
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                head = k[:7]
                if 'backbone.' in k:
                    tmp_name =k.replace('backbone.', '')
                else:
                    if "linear" in k:
                        continue
                    tmp_name = k
                new_state_dict[tmp_name] = v
            model.load_state_dict(new_state_dict, strict=False)
    model=model.cuda()
    # 对参数分组 ，指定不同学习率， 但是这里看起来逻辑错误
    ignored_params = list(map(id, model.parameters()))
    base_params = filter(lambda p: id(p) not in ignored_params, model.parameters())
    lr=0.1

    if weights:
        state_dict = torch.load(weights)

        try:
            model.load_state_dict(state_dict)
        except Exception as e:
            print("load not good ,start strict=False ")
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                head = k[:7]
                if head == 'module.':
                    tmp_name = k[7:]  # remove `module.`
                else:
                    if "linear" in k:
                        continue
                    tmp_name = k
                new_state_dict[tmp_name] = v

            model.load_state_dict(new_state_dict,strict=False)

        lr=0.01

    optimizer_ft = AdaBelief([
        {'params': base_params, 'lr': lr},
        {'params': model.parameters(), 'lr': lr},
    ], betas=(0.5, 0.999), eps=1e-8)
    m_t_max=60
    optimizer_ft = AdamW(model.parameters(), lr=0.001, betas=(0.9, 0.999))
    # optimizer_ft = optim.SGD([
    #     {'params': base_params, 'lr': lr},
    #     {'params': model.parameters(), 'lr': lr},
    # ], weight_decay=5e-4, momentum=0.9, nesterov=True)

    m_lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer_ft, T_max=m_t_max)
    # exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=40, gamma=0.1)

    # dir_name = './models_tiaosheng_0615/'+name
    # os.makedirs(dir_name, exist_ok=True)

    model = train_model(model, optimizer_ft, m_lr_scheduler,
                    num_epochs=300)