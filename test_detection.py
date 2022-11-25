import argparse
import json
import os
import sys
from pathlib import Path
from threading import Thread
import pandas as pd
from PIL import Image
from clearml import Task
import numpy as np
import torch
import yaml
from tqdm import tqdm
from tidecv import TIDE, datasets, Data

import utils.clearml_task
from utils.clearml_task import initialize_clearml_task
from utils.common import get_data_from_yaml


FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
sys.path.append(str(os.path.join(ROOT, 'yolov7')))
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from utils.datasets import create_dataloader
from utils.general import check_file, box_iou, scale_coords, xyxy2xywh, xywh2xyxy, set_logging, increment_path, colorstr
from utils.metrics import ap_per_class, ConfusionMatrix
from utils.plots import plot_images, output_to_target, plot_study_txt

def clearml_init_task(opt):
    config = get_data_from_yaml(opt.config)
    initialize_clearml_task(clearml_arguments=config.train.clearml_arguments, s3_config=config.s3,
                            remotely=opt.remotely, no_queue=opt.no_queue,
                            task_type=Task.TaskTypes.training)
    utils.clearml_task.clearml_task.connect(config.to_dict())
    utils.clearml_task.clearml_task.connect(opt)


def test(data,
         detections,
         batch_size=32,
         imgsz=1024,
         single_cls=False,
         verbose=False,
         save_dir=Path(''),  # for saving images
         plots=True):



    tide = TIDE()
    
    gt_data = Data('ground truth')
    det_data = Data('detections')

    # Set device

    set_logging()
    device = "cpu" #select_device(opt.device, batch_size=batch_size)

    # Directories
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
    save_dir.mkdir(parents=True, exist_ok=True)  # make dir

    with open(data) as f:
        data = yaml.load(f, Loader=yaml.SafeLoader)
    #check_dataset(data)  # check
    nc = 1 if single_cls else int(data['nc'])  # number of classes
    iouv = torch.linspace(0.5, 0.95, 10).to(device)  # iou vector for mAP@0.5:0.95
    niou = iouv.numel()

    # Dataloader
    task = opt.task if opt.task in ('val', 'test') else 'val'  # path to train/val/test images
    dataloader_det = create_dataloader(data[task], imgsz, batch_size, 32, opt=opt, pad=0.5, rect=True, labels_dir=detections,
                                       prefix=colorstr(f'{task}: '))[0]
    dataloader_grt = create_dataloader(data[task], imgsz, batch_size, 32, opt=opt, pad=0.5, rect=True, labels_dir="labels",
                                       prefix=colorstr(f'{task}: '))[0]
    seen = 0
    confusion_matrix = ConfusionMatrix(nc=nc)
    names = {k: v for k, v in enumerate(data["names"])}
    s = ('%20s' + '%12s' * 6) % ('Class', 'Images', 'Labels', 'P', 'R', 'mAP@.5', 'mAP@.5:.95')
    p, r, f1, mp, mr, map50, map, t0, t1 = 0., 0., 0., 0., 0., 0., 0., 0., 0.
    jdict, stats, ap, ap_class = [], [], [], []
    #pbar = tqdm(zip(dataloader_grt, dataloader_det), desc=s, bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
    nimg = 0
    for batch_i, (img, targets, paths, shapes) in enumerate(tqdm(dataloader_grt, desc=s)):
        if batch_i > 9:
            break
        _, netouts, _, _ = next(dataloader_det.iterator)

        for det in targets.numpy():
            gt_data.add_ground_truth(nimg+det[0], det[2], det[3:])
        for det in netouts.numpy():
            det_data.add_detection(nimg+det[0], det[2], det[1], det[3:])
        nimg += img.shape[0]

        nb, _, height, width = img.shape  # batch size, channels, height, width
        img = img.to(device, non_blocking=True)
        img = img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0

        targets = targets.to(device)
        targets = torch.hstack([targets[:, :1], targets[:, 2:]])
        targets[:, 2:] *= torch.Tensor([width, height, width, height]).to(device)  # to pixels

        netouts = netouts.to(device)
        netouts = torch.hstack([netouts[:, :1], netouts[:, 3:], netouts[:, 1:3]])
        netouts[:, 1:5] *= torch.Tensor([width, height, width, height]).to(device)
        netouts[:, 1:5] = xywh2xyxy(netouts[:, 1:5])
        netouts = [netouts[netouts[:, 0] == i, 1:] for i in range(img.shape[0])] 
              
        # Statistics per image
        for si, pred in enumerate(netouts):
            labels = targets[targets[:, 0] == si, 1:]
            nl = len(labels)
            tcls = labels[:, 0].tolist() if nl else []  # target class
            path = Path(paths[si])
            seen += 1

            if len(pred) == 0:
                if nl:
                    stats.append((torch.zeros(0, niou, dtype=torch.bool), torch.Tensor(), torch.Tensor(), tcls))
                continue

            # Predictions
            predn = pred.clone()
            scale_coords(img[si].shape[1:], predn[:, :4], shapes[si][0], shapes[si][1])  # native-space pred

            # Assign all predictions as incorrect
            correct = torch.zeros(pred.shape[0], niou, dtype=torch.bool, device=device)
            if nl:
                detected = []  # target indices
                tcls_tensor = labels[:, 0]

                # target boxes
                tbox = xywh2xyxy(labels[:, 1:5])
                scale_coords(img[si].shape[1:], tbox, shapes[si][0], shapes[si][1])  # native-space labels
                if plots:
                    confusion_matrix.process_batch(predn, torch.cat((labels[:, 0:1], tbox), 1))

                # Per target class
                for cls in torch.unique(tcls_tensor):
                    ti = (cls == tcls_tensor).nonzero(as_tuple=False).view(-1)  # prediction indices
                    pi = (cls == pred[:, 5]).nonzero(as_tuple=False).view(-1)  # target indices

                    # Search for detections
                    if pi.shape[0]:
                        # Prediction to target ious
                        ious, i = box_iou(predn[pi, :4], tbox[ti]).max(1)  # best ious, indices

                        # Append detections
                        detected_set = set()
                        for j in (ious > iouv[0]).nonzero(as_tuple=False):
                            d = ti[i[j]]  # detected target
                            if d.item() not in detected_set:
                                detected_set.add(d.item())
                                detected.append(d)
                                correct[pi[j]] = ious[j] > iouv  # iou_thres is 1xn
                                if len(detected) == nl:  # all targets already located in image
                                    break

            # Append statistics (correct, conf, pcls, tcls)
            stats.append((correct.cpu(), pred[:, 4].cpu(), pred[:, 5].cpu(), tcls))

        # Plot images
        if plots and batch_i < 3:
            f = save_dir / f'test_batch{batch_i}_labels.jpg'  # labels
            Thread(target=plot_images, args=(img, targets, paths, f, names), daemon=True).start()
            f = save_dir / f'test_batch{batch_i}_pred.jpg'  # predictions
            Thread(target=plot_images, args=(img, output_to_target(netouts), paths, f, names), daemon=True).start()

    # Compute statistics
    stats = [np.concatenate(x, 0) for x in zip(*stats)]  # to numpy
    if len(stats) and stats[0].any():
        p, r, ap, f1, ap_class = ap_per_class(*stats, plot=plots, save_dir=save_dir, names=names)
        ap50, ap = ap[:, 0], ap.mean(1)  # AP@0.5, AP@0.5:0.95
        mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
        nt = np.bincount(stats[3].astype(np.int64), minlength=nc)  # number of targets per class
    else:
        nt = torch.zeros(1)

    # Print results
    columns = ['Class', 'Images', 'Labels', 'P', 'R', 'mAP@.5', 'mAP@.5:.95']
    metrics_table = pd.DataFrame([], columns=columns)
    pf = '%20s' + '%12i' * 2 + '%12.3g' * 4  # print format
    print(pf % ('all', seen, nt.sum(), mp, mr, map50, map))
    #metrics_table.append({'Class': 'all', 'Images': seen, 'Labels': nt.sum(), 'P': mp, 'R': mr, 'mAP@.5': map50, 'mAP@.5:.95': map}, ignore_index=True, inplace=True)
    metrics_table = pd.concat([metrics_table, pd.DataFrame([['all', seen, nt.sum(), mp, mr, map50, map]], columns=columns)], ignore_index=True)

    # Print results per class
    if (verbose or nc < 50) and nc > 1 and len(stats):
        for i, c in enumerate(ap_class):
            print(pf % (names[c], seen, nt[c], p[i], r[i], ap50[i], ap[i]))
            metrics_table = pd.concat([metrics_table, pd.DataFrame([[names[c], seen, nt[c], p[i], r[i], ap50[i], ap[i]]], columns=columns)], ignore_index=True)
    utils.clearml_task.clearml_logger.report_table("Result metrics", "AP", 0, metrics_table)

    # TIDE metrics
    f = save_dir / f'tide_metrics'
    tide.evaluate(gt_data, det_data, mode=TIDE.BOX)
    tide.summarize()
    tide.plot(f)
    utils.clearml_task.clearml_logger.report_image("manual title", "TIDE", 0, f/"detections_bbox_summary.png")

    # Plots
    if plots:
        confusion_matrix.plot(save_dir=save_dir, names=list(names.values()))

    # Return results
    print(f"Results saved to {save_dir}")
    maps = np.zeros(nc) + map
    for i, c in enumerate(ap_class):
        maps[c] = ap[i]
    return (mp, mr, map50, map, maps)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='test.py')
    parser.add_argument('--data', type=str, default='data/coco.yaml', help='*.data path')
    parser.add_argument('--detections', type=str, help='dir to labels from detector. haveto be placed near labels folder, example - labels-from-model')
    parser.add_argument('--batch-size', type=int, default=32, help='size of each image batch')
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.001, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.65, help='IOU threshold for NMS')
    parser.add_argument('--task', default='test', help='train, val, test, speed or study')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--single-cls', action='store_true', help='treat as single-class dataset')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--verbose', action='store_true', help='report mAP by class')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-hybrid', action='store_true', help='save label+prediction hybrid results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--project', default='runs/test', help='save to project/name')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--config', type=str, default='configs/config.yaml', help='Path to a config file.')
    parser.add_argument('--remotely', action='store_true', help='Execute ClearML task remotely.')
    parser.add_argument('--no-queue', action='store_true', help='Choose to send to queue or not.')

    opt = parser.parse_args()
    opt.data = check_file(opt.data)  # check file
    print(opt)
    #check_requirements()

    clearml_init_task(opt)


    if opt.task in ('train', 'val', 'test'):  # run normally
        test(data=opt.data,
             detections=opt.detections,
             batch_size=opt.batch_size,
             verbose=opt.verbose,
             )
