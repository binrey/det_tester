import argparse
import json
import os
import sys
from pathlib import Path, PosixPath
from threading import Thread
import pandas as pd
from PIL import Image
from clearml import Task
import numpy as np
import torch
import yaml
import shutil
from tqdm import tqdm
from tidecv import TIDE, Data
import matplotlib.pyplot as plt

import utils.clearml_task
from utils.clearml_task import initialize_clearml_task
from utils.common import get_data_from_yaml
from globox import AnnotationSet

from utils.datasets import create_dataloader
from utils.general import check_file, box_iou, scale_coords, xyxy2xywh, xywh2xyxy, set_logging, increment_path, colorstr, ImageLoader
from utils.metrics import ap_per_class, ConfusionMatrix
from utils.plots import plot_images, output_to_target, plot_study_txt


def clearml_init_task(opt):
    s3config = get_data_from_yaml(opt.s3config)
    initialize_clearml_task(project_name=opt.project, 
                            task_name=opt.name, 
                            tags=opt.tags, 
                            s3_config=s3config, 
                            task_type=Task.TaskTypes.testing)
    utils.clearml_task.clearml_task.connect(s3config.to_dict())
    utils.clearml_task.clearml_task.connect(opt)


def test(imgs_path:PosixPath,
         grounds_annfile,
         detections_annfile,
         batch_size=32,
         nplots=10,
         imgsz=1024,
         log2clearml=False,
         save_dir=Path(""),  # for saving images
         plots=True):
    
    tide = TIDE()
    gt_data = Data("ground truth")
    det_data = Data("detections")

    # Directories
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
    save_dir.mkdir(parents=True, exist_ok=True)  # make dir

    # with open(data) as f:
    #     data = yaml.load(f, Loader=yaml.SafeLoader)
    annset_gt = AnnotationSet.from_coco(grounds_annfile)
    annset_pr = AnnotationSet.from_coco(detections_annfile)

    # check_dataset(data)  # check
    nc = len(annset_gt._id_to_label)
    lab2id = {v:k for k, v in annset_gt._id_to_label.items()}
    iouv = np.linspace(0.5, 0.95, 10) # iou vector for mAP@0.5:0.95
    niou = len(iouv)

    seen = 0
    confusion_matrix = ConfusionMatrix(nc=nc, log2clearml=log2clearml)
    names = annset_gt._id_to_label
    s = ("%20s" + "%12s" * 6) % ("Class", "Images", "Labels", "P", "R", "mAP@.5", "mAP@.5:.95")
    p, r, f1, mp, mr, map50, map, t0, t1 = 0., 0., 0., 0., 0., 0., 0., 0., 0.
    jdict, stats, ap, ap_class = [], [], [], []
    # pbar = tqdm(zip(dataloader_grt, dataloader_det), desc=s, bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}")
    stats_per_image = {}
    imgs4plot, gt_labs4plot, pr_labs4plot, gt_bboxes4plot, pr_bboxes4plot, confs4plot, ids4plot, maps4plot = np.zeros((batch_size, 3, imgsz, imgsz)), [], [], [], [], [], [], []
    for nimg, (img_id, ann_gt) in enumerate(tqdm(annset_gt.items())):
        if img_id not in annset_pr._annotations.keys():
            continue
        ann_pr = annset_pr[img_id]
        bboxes_gt = ann_gt.boxes
        bboxes_pr = ann_pr.boxes

        for bbox in bboxes_gt:
            gt_data.add_ground_truth(nimg, lab2id[bbox.label], np.array(bbox.xywh))
        for bbox in bboxes_pr:
            det_data.add_detection(nimg, lab2id[bbox.label], bbox.confidence, np.array(bbox.xywh)+1)


        labels = np.zeros((len(bboxes_gt), 5))
        for i, bbox in enumerate(bboxes_gt):
            labels[i, 0], labels[i, 1:] = lab2id[bbox.label], bbox.ltrb
        pred = np.zeros((len(bboxes_pr), 6))
        for i, bbox in enumerate(bboxes_pr):
            pred[i, :4], pred[i, 4], pred[i, 5] = bbox.ltrb, bbox.confidence, lab2id[bbox.label]

        nl = len(labels)
        tcls = labels[:, 0].tolist() if nl else []  # target class
        # path = Path(paths[si])
        seen += 1

        if len(pred) == 0:
            if nl:
                stats.append((torch.zeros(0, niou, dtype=torch.bool), torch.Tensor(), torch.Tensor(), tcls))
            continue

        # Predictions
        predn = pred.copy()
        # scale_coords(img[si].shape[1:], predn[:, :4], shapes[si][0], shapes[si][1])  # native-space pred

        # Assign all predictions as incorrect
        correct = np.zeros((pred.shape[0], niou))
        if nl:
            detected = []  # target indices
            tcls_tensor = labels[:, 0]

            # target boxes
            tbox = labels[:, 1:5]
            # scale_coords(img[si].shape[1:], tbox, shapes[si][0], shapes[si][1])  # native-space labels
            if plots:
                confusion_matrix.process_batch(predn, np.concatenate([labels[:, 0:1], tbox], 1))

            # Per target class
            for cls in np.unique(tcls_tensor):
                ti = np.array((cls == tcls_tensor).nonzero())[0]  # prediction indices
                pi = np.array((cls == pred[:, 5]).nonzero())[0]  # target indices

                # Search for detections
                if pi.shape[0]:
                    # Prediction to target ious
                    ious, i = box_iou(predn[pi, :4], tbox[ti]).max(1)  # best ious, indices
                    ious = ious.numpy()  # TODO
                    # Append detections
                    detected_set = set()
                    for j in (ious >= iouv[0]).nonzero()[0]:
                        d = ti[i[j]]  # detected target
                        if d.item() not in detected_set:
                            detected_set.add(d.item())
                            detected.append(d)
                            correct[pi[j]] = ious[j] > iouv  # iou_thres is 1xn
                            if len(detected) == nl:  # all targets already located in image
                                break

        # Append statistics (correct, conf, pcls, tcls)
        stats.append((correct, pred[:, 4], pred[:, 5], tcls))

        stats4img = [np.concatenate(x, 0) for x in zip(*stats[-1:])]  # to numpy
        if len(stats4img) and stats4img[0].any():
            p, r, ap, f1, ap_class = ap_per_class(*stats4img, plot=False, save_dir=None, names=names, log2clearml=False)
            ap = [ap[:, 0].mean(), ap[:, :].mean()]  # AP@0.5
        else:
            ap = [0, 0] if len(tcls) else [1, 1]
        stats_per_image.update({img_id: ap})

        # Plot images
        nplot = int((nimg+1)/batch_size)
        if plots and nplot <= nplots:
            ids4plot.append(img_id)
            maps4plot.append(f"{img_id} APs,%: {stats_per_image[img_id][0]*100:3.0f}, {stats_per_image[img_id][1]*100:3.0f}")
            img_loader = ImageLoader(imgsz)
            img = img_loader.load_image(imgs_path / img_id).transpose((2, 0, 1))
            gt_bboxes4plot.append(img_loader.correct_bboxes(labels[:, 1:]))
            pr_bboxes4plot.append(img_loader.correct_bboxes(pred[:, :4]))
            gt_labs4plot.append(labels[:, 0])
            pr_labs4plot.append(pred[:, 5])
            confs4plot.append(pred[:, 4])
            imgs4plot[len(ids4plot)-1] = img
            if len(ids4plot) == batch_size:
                f = save_dir / f"batch{nplot}_gtruth.jpg"  # labels
                plot_images(imgs4plot, gt_bboxes4plot, gt_labs4plot, None, ids4plot, f, names, log2clearml=log2clearml)        
                # Thread(target=plot_images, args=(imgs4plot, gt_bboxes4plot, gt_labs4plot, None, ids4plot, f, names, log2clearml=log2clearml), daemon=True).start()
                f = save_dir / f"batch{nplot}_pred.jpg"  # predictions
                plot_images(imgs4plot, pr_bboxes4plot, pr_labs4plot, confs4plot, maps4plot, f, names, log2clearml=log2clearml)        
                # Thread(target=plot_images, args=(imgs4plot, pr_bboxes4plot, pr_labs4plot, confs4plot, ids4plot, f, names, log2clearml=log2clearml), daemon=True).start()
                imgs4plot, gt_labs4plot, pr_labs4plot, gt_bboxes4plot, pr_bboxes4plot, confs4plot, ids4plot, maps4plot = np.zeros((batch_size, 3, imgsz, imgsz)), [], [], [], [], [], [], []

    # Compute statistics
    stats = [np.concatenate(x, 0) for x in zip(*stats)]  # to numpy
    if len(stats) and stats[0].any():
        p, r, ap, f1, ap_class = ap_per_class(*stats, plot=plots, save_dir=save_dir, names=names, log2clearml=log2clearml)
        ap50, ap = ap[:, 0], ap.mean(1)  # AP@0.5, AP@0.5:0.95
        mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
        nt = np.bincount(stats[3].astype(np.int64), minlength=nc)  # number of targets per class
    else:
        nt = torch.zeros(1)

    # Print results
    columns = ["Class", "Images", "Labels", "P", "R", "mAP@.5", "mAP@.5:.95"]
    metrics_table = pd.DataFrame([], columns=columns)
    pf = "%20s" + "%12i" * 2 + "%12.3g" * 4  # print format
    print(pf % ("all", seen, nt.sum(), mp, mr, map50, map))
    metrics_table = pd.concat(
        [metrics_table, pd.DataFrame([["all", seen, nt.sum(), mp, mr, map50, map]], columns=columns)],
        ignore_index=True)

    # Print results per class
    if nc < 50 and nc > 1 and len(stats):
        for i, c in enumerate(ap_class):
            print(pf % (names[c], seen, nt[c], p[i], r[i], ap50[i], ap[i]))
            metrics_table = pd.concat(
                [metrics_table, pd.DataFrame([[names[c], seen, nt[c], p[i], r[i], ap50[i], ap[i]]], columns=columns)],
                ignore_index=True)
    if log2clearml:
        utils.clearml_task.clearml_logger.report_table("General metrics", "mAPs table", 0, metrics_table)

    with open(save_dir / "metrics.txt", "w") as mf:
        mf.write(metrics_table.to_string())

    # TIDE metrics
    try:
        f = save_dir / f"tide_metrics"
        tide.evaluate(gt_data, det_data, mode=TIDE.BOX)
        tide.summarize()
        tide.plot(f)
        shutil.move(f / "detections_bbox_summary.png", f.parent / "TIDE.png")
        shutil.rmtree(f)
        f = f.parent / "TIDE.png"
        fig = plt.figure(figsize=(12, 12), tight_layout=True)
        plt.imshow(np.array(Image.open(f)))
        plt.axis("off")
        if log2clearml:
            utils.clearml_task.clearml_logger.report_matplotlib_figure("General metrics", "TIDE", fig, report_interactive=False)
    except ZeroDivisionError:
        pass

    # Plots
    if plots:
        confusion_matrix.plot(save_dir=save_dir, names=list(names.values()))

    json.dump(stats_per_image, open(save_dir / "stats.json", "w"))

    # Return results
    print(f"Results saved to {save_dir}")
    maps = np.zeros(nc) + map
    for i, c in enumerate(ap_class):
        maps[c] = ap[i]
    return (mp, mr, map50, map, maps)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="test.py")
    parser.add_argument("--data_path", type=str, help="*.data path")
    parser.add_argument("--grounds", type=str, help="ground annotations coco format")    
    parser.add_argument("--predicts", type=str, help="predictions in coco format")
    parser.add_argument("--batch-size", type=int, default=32, help="size of each image batch")
    parser.add_argument("--img-size", type=int, default=640, help="inference size (pixels)")
    parser.add_argument("--project", default="runs/test", help="save to project/name")
    parser.add_argument("--name", default="exp", help="save to project/name")
    parser.add_argument("--tags", type=str, nargs='+', default=[], help="clearml task tags")
    parser.add_argument("--s3config", type=str, default="configs/config.yaml", help="Path to a config file.")
    parser.add_argument("--clearml", action="store_true", help="use clearml logging")
    parser.add_argument("--exist-ok", action="store_true", help="existing project/name ok, do not increment")

    opt = parser.parse_args()
    print(opt)
    if opt.clearml:
        clearml_init_task(opt)
    test(Path(opt.data_path), opt.grounds, opt.predicts, batch_size=opt.batch_size, log2clearml=opt.clearml)
