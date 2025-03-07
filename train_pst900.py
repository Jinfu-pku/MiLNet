import os
import shutil
import json
import time

# from apex import amp
from torch.cuda import amp
import copy

import numpy as np
import torch.distributed as dist

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.optim.lr_scheduler import LambdaLR, StepLR
from torch.utils.data import DataLoader

from toolbox import MscCrossEntropyLoss
from toolbox import get_dataset
from toolbox import get_logger
from toolbox import get_model
from toolbox.metricsm import averageMeter, runningScore
from toolbox import ClassWeight, save_ckpt
from toolbox import Ranger
from toolbox import setup_seed
from toolbox import load_ckpt
from toolbox import group_weight_decay
from toolbox import CrossEntropyLoss2d, CrossEntropyLoss2dLabelSmooth, ProbOhemCrossEntropy2d, FocalLoss2d, \
    LovaszSoftmax, LDAMLoss

from toolbox.segment_anything import SamPredictor, sam_model_registry

from torch.multiprocessing import set_start_method

setup_seed(33)


class eeemodelLoss(nn.Module):

    def __init__(self, class_weight=None, ignore_index=-100, reduction='mean'):
        super(eeemodelLoss, self).__init__()

        self.class_weight_semantic = torch.from_numpy(np.array(
            [1.5105, 16.6591, 29.4238, 34.6315, 40.0845, 41.4357, 47.9794, 45.3725, 44.9000])).float()
        self.class_weight_binary = torch.from_numpy(np.array([1.5121, 10.2388])).float()
        self.class_weight_boundary = torch.from_numpy(np.array([1.4459, 23.7228])).float()

        self.LovaszSoftmax = LovaszSoftmax()

        self.cross_entropy = nn.CrossEntropyLoss()
        self.semantic_loss_weight = nn.CrossEntropyLoss(weight=self.class_weight_semantic)
        self.semantic_loss = nn.CrossEntropyLoss()

        # self.binary_loss = nn.CrossEntropyLoss(weight=self.class_weight_binary)
        self.binary_loss = nn.CrossEntropyLoss()
        # self.boundary_loss = nn.CrossEntropyLoss(weight=self.class_weight_boundary)
        self.boundary_loss = nn.CrossEntropyLoss()  # 为了避免出现weight的错误

    def forward(self, inputs, targets):
        # semantic_out, boundary_out, l1_out, l2_out, l3_out, l4_out, l5_out, = inputs
        # semantic_gt, boundary_gt, l1_gt, l2_gt, l3_gt, l4_gt, l5_gt = targets

        # semantic_out, l1_out, l2_out, l3_out, l4_out, l5_out, = inputs
        # semantic_gt, l1_gt, l2_gt, l3_gt, l4_gt, l5_gt = targets

        out, out_layer1, out_layer2, out_layer3, out_layer4 = inputs
        semantic_gt, l_gt1, l_gt2, l_gt3, l_gt4 = targets

        # loss_crossentropy = F.cross_entropy(inputs, targets, weight=None)
        # loss_crossentropy_with_classweight = F.cross_entropy(inputs, targets, weight=self.class_weight)
        # loss_lovasz = self.LovaszSoftmax(inputs, targets)
        #
        # alpha = 0.8
        # # loss = loss_crossentropy_with_classweight + loss_lovasz
        # loss = loss_lovasz * alpha + loss_crossentropy_with_classweight * (1 - alpha)
        # loss1 = self.cross_entropy(semantic_out, semantic_gt)

        loss_pre = self.LovaszSoftmax(out, semantic_gt)  # decoder的输出用 语义标签

        loss_sv1 = self.boundary_loss(out_layer1, l_gt1)  # 前两层用 bound 边界标签
        loss_sv2 = self.binary_loss(out_layer2, l_gt2)

        loss_sv3 = self.binary_loss(out_layer3, l_gt3)  # 后两层用 binary 前背景标签
        loss_sv4 = self.boundary_loss(out_layer4, l_gt4)

        loss = 2 * loss_pre + loss_sv1 + loss_sv2 + loss_sv3 + loss_sv4

        return loss


def run(args):
    torch.cuda.set_device(args.cuda)
    with open(args.config, 'r') as fp:
        cfg = json.load(fp)

    logdir = f'run/{time.strftime("%Y-%m-%d-%H-%M")}({cfg["dataset"]}-{cfg["model_name"]})'
    if not os.path.exists(logdir):
        os.makedirs(logdir)
    shutil.copy(args.config, logdir)

    logger = get_logger(logdir)
    logger.info(f'Conf | use logdir {logdir}')

    device = torch.device(f'cuda:{args.cuda}')

    # model**************************************

    model = get_model(cfg)
    model.to(device)

    sam = sam_model_registry["vit_h"](
        checkpoint="/home/liujf/MMSMCNet-main (1125)/toolbox/segment_anything/sam_vit_h_4b8939.pth")
    sam.to(device)
    predictor = SamPredictor(sam)

    # dataloader
    trainset, testset = get_dataset(cfg)

    train_loader = DataLoader(trainset, batch_size=cfg['ims_per_gpu'], shuffle=True, num_workers=cfg['num_workers'],
                              pin_memory=True, drop_last=True)  # 出现 pin_memory的相关错误，将它改成了pin_memory=False, 由于多进程加载数据，有tensor
    test_loader = DataLoader(testset, batch_size=cfg['ims_per_gpu'], shuffle=False, num_workers=cfg['num_workers'],
                             pin_memory=True, drop_last=True)

    params_list = model.parameters()
    optimizer = Ranger(params_list, lr=cfg['lr_start'], weight_decay=cfg['weight_decay'])
    scheduler = LambdaLR(optimizer, lr_lambda=lambda ep: (1 - ep / cfg['epochs']) ** 0.9)
    Scaler = amp.GradScaler()

    train_criterion = eeemodelLoss().to(device)
    criterion = nn.CrossEntropyLoss().to(device)

    soft_loss = nn.KLDivLoss(reduction='batchmean').to(device)
    # soft_loss = nn.MSELoss().to(device)

    # # 平局损失计算器
    train_loss_meter = averageMeter()
    test_loss_meter = averageMeter()

    ## 查看loss细节
    loss_hard_meter = averageMeter()
    loss_soft_meter = averageMeter()

    running_metrics_test = runningScore(cfg['n_classes'], ignore_index=cfg['id_unlabel'])
    best_test = 0
    best_miou = 0.85

    Temp = 2  # 蒸馏温度
    alpha = 0.5
    beta = 0.4

    # 每个epoch迭代循环
    for ep in range(cfg['epochs']):

        # training
        model.train()
        train_loss_meter.reset()  # 重置用于跟踪测试损失的对象，以便在每个测试周期开始时损失的记录从头开始
        for i, sample in enumerate(train_loader):
            optimizer.zero_grad()  # 梯度清零

            ################### train edit #######################
            if cfg['inputs'] == 'rgb':
                image = sample['image'].to(device)
                label = sample['label'].to(device)
                targets = label
                predict = model(image)
                # print(predict[0].shape)
            else:
                image = sample['image'].to(device)
                depth = sample['depth'].to(device)
                label = sample['label'].to(device)

                sam_input = sample['sam_input'].to(device)

                s_label1 = sample['bound4'].squeeze(1).to(device)
                s_label2 = sample['binary8'].squeeze(1).to(device)
                s_label3 = sample['binary8'].squeeze(1).to(device)
                s_label4 = sample['bound4'].squeeze(1).to(device)

                targets = [label, s_label1, s_label2, s_label3, s_label4]
                # img = image.squeeze(0).permute(1, 2, 0).detach().numpy().astype(np.uint8) # 匹配sam的输入

                # predict = model(image)
            with amp.autocast():
                teachers = []
                for i in range(cfg['ims_per_gpu']):
                    # 从当前 batch 中取出第 i 张图片
                    current_image = sam_input[i:i + 1]  # 保留第一个维度，即将第 i 张图片提取出来
                    img = current_image.squeeze(0).permute(1, 2, 0).cpu().detach().numpy().astype(np.uint8)  # 匹配sam的输入
                    with torch.no_grad():
                        # sam.val()         ????????
                        predictor.set_image(img)
                        image_embedding = predictor.get_image_embedding()  # (1, 256, 64, 64)
                        teacher = image_embedding[:, :, :48, :]  # (1, 256, 48, 64)
                        teachers.append(teacher)  # 将当前循环中的 teacher 加入列表
                teachers_batch = torch.stack(teachers, dim=0)

                pre_out, lossT, pre_layer1, pre_layer2, pre_layer3, pre_layer4, f_KD = model(image, depth)

                predict = [pre_out, pre_layer1, pre_layer2, pre_layer3, pre_layer4]

                loss_hard = train_criterion(predict, targets) + lossT  # student loss

                student = f_KD

                s = student.reshape(student.shape[0], -1)
                t = teachers_batch.reshape(teachers_batch.shape[0], -1)

                loss_soft = Temp * Temp * soft_loss(F.log_softmax(s / Temp),
                                                    F.softmax(t / Temp))  # distillation loss KL散度

                loss = alpha * loss_hard + (1 - alpha) * loss_soft
            ####################################################


            Scaler.scale(loss).backward()

            Scaler.step(optimizer)

            Scaler.update()

            train_loss_meter.update(loss.item())  # 计算平均损失，并更新。loss.item():取张量loss的标量值

            # loss细节

            loss_hard_meter.update(loss_hard.item())
            loss_soft_meter.update(loss_soft.item())

        scheduler.step(ep)

        # test
        with torch.no_grad():
            model.eval()  # 告诉我们的网络，这个阶段是用来测试的，于是模型的参数在该阶段不进行更新
            running_metrics_test.reset()  # 每个测试周期的test阶段，重新计算指标
            test_loss_meter.reset()  # 重置用于跟踪测试损失的对象，以便在每个测试周期开始时损失的记录从头开始
            for i, sample in enumerate(test_loader):
                if cfg['inputs'] == 'rgb':
                    image = sample['image'].to(device)
                    label = sample['label'].to(device)
                    predict = model(image)
                else:
                    image = sample['image'].to(device)
                    depth = sample['depth'].to(device)
                    label = sample['label'].to(device)

                    predict = model(image, depth)

                loss = criterion(predict, label)
                test_loss_meter.update(loss.item())  # 计算平均损失，并更新。loss.item():取张量loss的标量值

                # print("predict:", predict.shape)
                # argmax
                predict = predict.max(1)[1].cpu().numpy()  # predict.max(1): 对张量 predict 沿着维度 1 （即通道）进行最大值操作。这将返回一个包含最大值和对应索引的元组 (max_values, indices)。元组中的成员均为（B，H，W）的张量

                # [1]: 选择元组中的索引部分，即得到最大值的索引。（即最大值是哪个通道） argmax最终的结果得到一个（B，h，w）的张量。如下图所示（一批处理B个图像，下图为之一）

                #                      2000000000000
                #                      2000088000000
                #                      2000088000000
                #                      2000000000000
                #                      2000000111000
                #                      2000000111000

                label = label.cpu().numpy()
                # print("label,predict:", label.shape, predict.shape)
                running_metrics_test.update(label, predict)


        train_loss = train_loss_meter.val  # 打印当先损失
        test_loss = test_loss_meter.val
        # test_loss2 = test_loss_meter2.avg

        h_loss = loss_hard_meter.val  # 打印当前损失细节
        s_loss = loss_soft_meter.val

        test_macc = running_metrics_test.get_scores()[0]["class_acc: "]  # 类平均准确率
        test_miou = running_metrics_test.get_scores()[0]["mIou: "]  # 交并比
        test_avg = (test_macc + test_miou) / 2
        # # 第二个测试集
        # test_macc2 = running_metrics_test2.get_scores()[0]["class_acc: "]
        # test_miou2 = running_metrics_test2.get_scores()[0]["mIou: "]
        # test_avg2 = (test_macc2 + test_miou2) / 2

        # 每轮训练结束后打印结果
        logger.info(f'Iter | [{ep + 1:3d}/{cfg["epochs"]}] '
                    f'loss={train_loss:.3f}/{test_loss:.3f}, '
                    f'loss_hard={h_loss:.3f}, '
                    f'loss_soft={s_loss:.3f}, '

                    f'mPA={test_macc:.3f}, '
                    f'miou={test_miou:.3f}, '
                    f'avg={test_avg:.3f}')
        # logger.info(f'Iter | [{ep + 1:3d}/{cfg["epochs"]}] '
        #             f'loss={train_loss:.3f}/{test_loss:.3f}/{test_loss2:.3f}, '
        #             f'mPA={test_macc:.3f}/{test_macc2:.3f}, '
        #             f'miou={test_miou:.3f}/{test_miou2:.3f}, '
        #             f'avg={test_avg:.3f}/{test_avg2:.3f}')
        if test_miou > best_miou:
            best_miou = test_miou
            save_ckpt(logdir, model)
            print(test_miou)


if __name__ == '__main__':
    # set_start_method('spawn')           ## 为了解决mutiprocess的相关问题

    import argparse

    parser = argparse.ArgumentParser(description="config")
    parser.add_argument("--config", type=str, default="configs/MiLNet_pst.json", help="Configuration file to use")
    parser.add_argument("--opt_level", type=str, default='O1')
    parser.add_argument("--inputs", type=str.lower, default='rgb', choices=['rgb', 'rgbd'])
    parser.add_argument("--resume", type=str, default='',
                        help="use this file to load last checkpoint for continuing training")
    parser.add_argument("--cuda", type=int, default=5, help="set cuda device id")
    parser.add_argument("--备注", type=str, default="", help="记录配置和对照组")

    args = parser.parse_args()

    run(args)
