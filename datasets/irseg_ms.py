import os
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split

import torch
import torch.utils.data as data
from torchvision import transforms
from toolbox.datasets.augmentations import Reshape, Resize, Compose, ColorJitter, RandomHorizontalFlip, RandomCrop, RandomScale, \
    RandomRotation

import torchvision.transforms.functional as F
from torchvision.transforms.functional import InterpolationMode

class IRSeg(data.Dataset):

    def __init__(self, cfg, mode='trainval', do_aug=True):

        assert mode in ['train', 'val', 'trainval', 'test', 'test_day', 'test_night'], f'{mode} not support.'
        self.mode = mode

        ## pre-processing
        self.im_to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

        self.dp_to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.449, 0.449, 0.449], [0.226, 0.226, 0.226]),
        ])

        self.root = cfg['root']
        self.n_classes = cfg['n_classes']

        scale_range = tuple(float(i) for i in cfg['scales_range'].split(' '))
        crop_size = tuple(int(i) for i in cfg['crop_size'].split(' '))

        #######
        l1_size = tuple(int(i) for i in cfg['l1_size'].split(' '))
        l2_size = tuple(int(i) for i in cfg['l2_size'].split(' '))
        l3_size = tuple(int(i) for i in cfg['l3_size'].split(' '))
        l4_size = tuple(int(i) for i in cfg['l4_size'].split(' '))
        l5_size = tuple(int(i) for i in cfg['l5_size'].split(' '))

        #######

        self.aug = Compose([
            ColorJitter(
                brightness=cfg['brightness'],
                contrast=cfg['contrast'],
                saturation=cfg['saturation']),
            RandomHorizontalFlip(cfg['p']),
            RandomScale(scale_range),
            RandomCrop(crop_size, pad_if_needed=True)
        ])

        # self.val_resize = Resize(crop_size)

        ##########
        self.l1_sv = Reshape(l1_size)
        self.l2_sv = Reshape(l2_size)
        self.l3_sv = Reshape(l3_size)
        self.l4_sv = Reshape(l4_size)
        self.l5_sv = Reshape(l5_size)

        ##########

        self.mode = mode
        self.do_aug = do_aug

        if cfg['class_weight'] == 'enet':
            self.class_weight = np.array(
                [1.5105, 16.6591, 29.4238, 34.6315, 40.0845, 41.4357, 47.9794, 45.3725, 44.9000])
            self.binary_class_weight = np.array([1.5121, 10.2388])
        elif cfg['class_weight'] == 'median_freq_balancing':
            self.class_weight = np.array(
                [0.0118, 0.2378, 0.7091, 1.0000, 1.9267, 1.5433, 0.9057, 3.2556, 1.0686])
            self.binary_class_weight = np.array([0.5454, 6.0061])
        else:
            raise (f"{cfg['class_weight']} not support.")

        with open(os.path.join(self.root, f'{mode}.txt'), 'r') as f:
            self.infos = f.readlines()

    def __len__(self):
        return len(self.infos)

    def __getitem__(self, index):
        image_path = self.infos[index].strip()

        # print(image_path)

        # image = Image.open(os.path.join(self.root, 'images', image_path + '.png'))
        # label = Image.open(os.path.join(self.root, 'labels', image_path + '.png'))

        image = Image.open(os.path.join(self.root, 'seperated_images', image_path + '_rgb.png'))
        depth = Image.open(os.path.join(self.root, 'seperated_images', image_path + '_th.png')).convert('RGB')
        label = Image.open(os.path.join(self.root, 'labels', image_path + '.png'))

        bound = Image.open(os.path.join(self.root, 'bound', image_path+'.png'))
        binary_label = Image.open(os.path.join(self.root, 'binary_labels', image_path + '.png'))
        # attention_map = Image.open(os.path.join(self.root, 'attention_map', image_path + '.png')
        ##########
        # l1_label = self.l1_sv(label)
        # l2_label = self.l2_sv(label)
        # l3_label = self.l3_sv(label)
        # l4_label = self.l4_sv(label)
        # l5_label = self.l5_sv(label)

        ##########

        # image = np.asarray(image)
        # im = image[:, :, :3]
        # dp = image[:, :, 3:]
        # dp = np.concatenate([dp, dp, dp], axis=2)
        # image = Image.fromarray(im)
        # depth = Image.fromarray(dp)

        sample = {
            'image': image,
            'depth': depth,
            'label': label,
            'bound': bound,
            'binary_label': binary_label,

            #######
            # 'attention_map': attention_map,
        }

        if self.mode in ['train', 'trainval'] and self.do_aug:  # 只对训练集增强
            sample = self.aug(sample)

        sample['l1_label'] = self.l1_sv(sample['label'])
        sample['l2_label'] = self.l2_sv(sample['label'])
        sample['l3_label'] = self.l3_sv(sample['label'])
        sample['l4_label'] = self.l4_sv(sample['label'])
        sample['l5_label'] = self.l5_sv(sample['label'])

        sample['image'] = self.im_to_tensor(sample['image'])
        sample['depth'] = self.dp_to_tensor(sample['depth'])
        sample['label'] = torch.from_numpy(np.asarray(sample['label'], dtype=np.int64)).long()
        sample['bound'] = torch.from_numpy(np.asarray(sample['bound'], dtype=np.int64) / 255.).long()
        sample['binary_label'] = torch.from_numpy(np.asarray(sample['binary_label'], dtype=np.int64) / 255.).long()
        # sample['attention_map'] = torch.from_numpy(np.asarray(sample['attention_map'], dtype=np.int64) / 255.).long()
        ########
        sample['l1_label'] = torch.from_numpy(np.asarray(sample['l1_label'], dtype=np.int64)).long()
        sample['l2_label'] = torch.from_numpy(np.asarray(sample['l2_label'], dtype=np.int64)).long()
        sample['l3_label'] = torch.from_numpy(np.asarray(sample['l3_label'], dtype=np.int64)).long()
        sample['l4_label'] = torch.from_numpy(np.asarray(sample['l4_label'], dtype=np.int64)).long()
        sample['l5_label'] = torch.from_numpy(np.asarray(sample['l5_label'], dtype=np.int64)).long()
        ########

        sample['label_path'] = image_path.strip().split('/')[-1] + '.png'  # 后期保存预测图时的文件名和label文件名一致
        return sample

    @property
    def cmap(self):
        return [
            (0, 0, 0),  # unlabelled
            (64, 0, 128),  # car
            (64, 64, 0),  # person
            (0, 128, 192),  # bike
            (0, 0, 192),  # curve
            (128, 128, 0),  # car_stop
            (64, 64, 128),  # guardrail
            (192, 128, 128),  # color_cone
            (192, 64, 0),  # bump
        ]


if __name__ == '__main__':
    import json

    path = '/home/ljf/Desktop/ljf/lxy/lxy_segmentation/lxy_code/Segmentation_final/configs/ex/rtgan.json'
    with open(path, 'r') as fp:
        cfg = json.load(fp)
    cfg['root'] = '/home/ljf/Desktop/ljf/lxy/lxy_segmentation/lxy_code/irseg'
    # dataset = IRSeg(cfg, mode='train')
    # from toolbox.utils import class_to_RGB
    # import matplotlib.pyplot as plt
    #
    # for i in range(len(dataset)):
    #     sample = dataset[i]
    #
    #     image = sample['image']
    #     depth = sample['depth']
    #     label = sample['binary_label']
    #     print(label)
    #
    #     image = image.numpy()
    #     image = image.transpose((1, 2, 0))
    #     image *= np.asarray([0.229, 0.224, 0.225])
    #     image += np.asarray([0.485, 0.456, 0.406])
    #
    #     depth = depth.numpy()
    #     depth = depth.transpose((1, 2, 0))
    #     depth *= np.asarray([0.226, 0.226, 0.226])
    #     depth += np.asarray([0.449, 0.449, 0.449])
    #
    #     label = label.numpy()
    #     # label = class_to_RGB(label, N=len(dataset.cmap), cmap=dataset.cmap)
    #
    #     plt.subplot('131')
    #     plt.imshow(image)
    #     plt.subplot('132')
    #     plt.imshow(depth)
    #     plt.subplot('133')
    #     plt.imshow(label)
    #
    #     plt.show()
    #
    #     if i == 5:
    #         break

    dataset = IRSeg(cfg, mode='train', do_aug=True)
    # print(len(dataset))
    # from toolbox.utils import ClassWeight
    #
    # train_loader = torch.utils.data.DataLoader(dataset, batch_size=cfg['ims_per_gpu'], shuffle=True,
    #                                            num_workers=cfg['num_workers'], pin_memory=True)
    # classweight = ClassWeight('enet')  # enet, median_freq_balancing
    # class_weight = classweight.get_weight(train_loader, 2)
    # class_weight = torch.from_numpy(class_weight).float()
    # # class_weight[cfg['id_unlabel']] = 0
    #
    # print(class_weight)
    # 1.5844, 6.7297

    # # median_freq_balancing
    # tensor([0.1048, 0.0839, 0.1959, 0.2881, 0.3726, 0.5093, 0.6121, 0.8205, 0.8337,
    #         0.7482, 0.6981, 0.8627, 1.1372, 0.8133, 1.1034, 1.4300, 1.0000, 0.9371,
    #         2.0121, 0.9904, 1.9158, 2.0197, 1.1565, 2.5478, 1.6299, 2.0030, 4.3178,
    #         2.5112, 0.9394, 4.5706, 0.8648, 2.5528, 4.1016, 2.2452, 4.7550, 6.3821,
    #         1.5921, 7.2107, 0.9008, 0.9054, 0.3956])

    # # enet
    # tensor([5.7207, 4.7615, 9.4599, 12.7489, 16.7200, 18.6128, 23.0818, 24.5198,
    #         25.3707, 23.7553, 27.1135, 25.0198, 30.0310, 26.6835, 32.8046, 33.5390,
    #         32.1986, 34.4739, 35.8989, 34.4298, 37.8402, 38.2055, 29.4286, 38.4170,
    #         40.4630, 40.5733, 42.4383, 42.2503, 42.1083, 43.2440, 42.5954, 44.7177,
    #         44.5846, 44.6458, 44.4627, 45.0588, 43.8649, 45.6229, 25.6861, 25.8775,
    #         15.8056])

    # bound classweight = [ 1.4459, 23.7228]
