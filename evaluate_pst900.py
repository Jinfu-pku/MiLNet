import os
import time
from tqdm import tqdm
from PIL import Image
import json

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

from toolbox import get_dataset
from toolbox import get_model
from toolbox.metricsm import averageMeter, runningScore
from toolbox import class_to_RGB, load_ckpt, save_ckpt

from toolbox.datasets.pst900 import PST900


def evaluate(logdir, save_predict=False, options=['val', 'test', 'test_day', 'test_night'], prefix=''):
    # 加载配置文件cfg
    cfg = None
    for file in os.listdir(logdir):
        if file.endswith('.json'):
            with open(os.path.join(logdir, file), 'r') as fp:
                cfg = json.load(fp)
    assert cfg is not None

    device = torch.device('cuda')

    loaders = []
    for opt in options:
        dataset = PST900(cfg, mode=opt)
        # dataset = PST900(cfg, mode=opt)
        loaders.append((opt, DataLoader(dataset, batch_size=1, shuffle=False, num_workers=cfg['num_workers'])))
        cmap = dataset.cmap

    model = get_model(cfg).to(device)

    # model.load_state_dict(torch.load(os.path.join(logdir, prefix+'model.pth'), map_location={'cuda:1': 'cuda:0'}))
    # save_ckpt('/home/dtrimina/Desktop/lxy/Segmentation_final/run', model)

    model = load_ckpt(logdir, model, prefix=prefix)

    running_metrics_val = runningScore(cfg['n_classes'], ignore_index=cfg['id_unlabel'])
    time_meter = averageMeter()

    save_path = os.path.join(logdir, 'predicts')
    if not os.path.exists(save_path) and save_predict:
        os.mkdir(save_path)

    for name, test_loader in loaders:
        running_metrics_val.reset()
        print('#'*50 + '    ' + name+prefix + '    ' + '#'*50)
        with torch.no_grad():
            model.eval()
            for i, sample in tqdm(enumerate(test_loader), total=len(test_loader)):

                time_start = time.time()

                if cfg['inputs'] == 'rgb':
                    image = sample['image'].to(device)
                    label = sample['label'].to(device)
                    predict = model(image)
                else:
                    image = sample['image'].to(device)
                    depth = sample['depth'].to(device)
                    label = sample['label'].to(device)
                    predict = model(image, depth)

                # predict = predict.max(1)[1].cpu().numpy()  # [1, h, w]
                predict = predict.max(1)[1].cpu().numpy()  # [1, h, w]
                label = label.cpu().numpy()
                # print("label, predict:", label.shape, predict.shape)
                running_metrics_val.update(label, predict)

                if save_predict:
                    predict = predict.squeeze(0)  # [1, h, w] -> [h, w]
                    predict = class_to_RGB(predict, N=len(cmap), cmap=cmap)  # 如果数据集没有给定cmap,使用默认cmap
                    predict = Image.fromarray(predict)
                    predict.save(os.path.join(save_path, sample['label_path'][0]))

        metrics = running_metrics_val.get_scores()
        print('overall metrics .....')
        for k, v in metrics[0].items():
            print(k, f'{v:.4f}')

        print('iou for each class .....')
        for k, v in metrics[1].items():
            print(k, f'{v:.4f}')
        print('acc for each class .....')
        for k, v in metrics[2].items():
            print(k, f'{v:.4f}')

        print(f'{metrics[0]["class_acc: "]:.4f}', f'{metrics[0]["mIou: "]:.4f}')



if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="evaluate")
    # parser.add_argument("--logdir", type=str, help="run logdir")
    # parser.add_argument("-s", type=bool, default=False, help="save predict or not")
    parser.add_argument("--logdir", default="run/run_pst900", type=str,
                        help="run logdir")
    parser.add_argument("-s", type=bool, default="run/run_pst900/predict",
                        help="save predict or not")
    args = parser.parse_args()

    # prefix option ['', 'best_val_', 'best_test_]
    # options=['test', 'test_day', 'test_night']
    evaluate(args.logdir, save_predict=args.s, options=['test'], prefix='')
    # evaluate(args.logdir, save_predict=args.s, options=['val'], prefix='')
    # evaluate(args.logdir, save_predict=args.s, options=['test_day'], prefix='')
    # evaluate(args.logdir, save_predict=args.s, options=['test_night'], prefix='')
    # msc_evaluate(args.logdir, save_predict=args.s)