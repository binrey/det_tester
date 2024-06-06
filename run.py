import json
import shutil
from pathlib import Path
from typing import List, Optional, Tuple, Union
from utils.dataloading import load_dataset

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from globox import AnnotationSet
from PIL import Image
from tqdm import tqdm
import logging

import utils.clearml_task
from utils.clearml_task import clearml_init_task
from utils.general import (ImageLoader, box_iou, increment_path)
from utils.metrics import ConfusionMatrix, ap_per_class
from utils.object_selector import ObjectSelector
from utils.plots import plot_data_stats, plot_images

# Set up logging
logging.basicConfig(level=logging.INFO)

# TODO
import os
os.environ["CURL_CA_BUNDLE"] = "/home/rybin-av/myCA.crt"
os.environ["REQUESTS_CA_BUNDLE"] = "/home/rybin-av/myCA.crt"

def run_test(imgs_path: str,
             grounds_annfile: str,
             detections_annfile: str,
             batch_size: int = 4,
             nplots: int = 10,
             imgsz: int = 1024,
             save_images: bool = True,
             log2clearml: bool = True,
             add_data_stats: bool = True,
             add_tide: bool = True,
             add_false_negatives: bool = True,
             project_name: str = "Testing",
             task_name: str = "example",
             task_comment: Optional[str] = None,
             tags: List[str] = []
             ) -> str:
    """
    Runs a test on the given images, ground truth annotations, and detections.

    Args:
        imgs_path: Path to the images.
        grounds_annfile: Path to the ground truth annotations.
        detections_annfile: Path to the detections.
        batch_size: Amount of images to plot in collage.
        nplots: Amount of collages.
        imgsz: Size of collage.
        save_images: Whether to save images.
        log2clearml: Whether to use ClearML for logging.
        add_data_stats: Whether to calculate and plot statistics of objects sizes.
        add_tide: Whether to calculate and plot TIDE metric.
        add_false_negatives: Whether to add false negatives examples.
        project_name: Project name in ClearML.
        task_name: Task name in ClearML.
        task_comment: Task comment.
        tags: ClearML task tags.

    Returns:
        str: Path to clearml log.
    """

    path2log = clearml_init_task(project_name=project_name, task_name=task_name, tags=tags, comment=task_comment) if log2clearml else ""

    if imgs_path is not None:
        imgs_path = Path(imgs_path)
    else:
        save_images = False

    if add_tide:
        from tidecv import TIDE, Data
        tide = TIDE()
        gt_data = Data("ground truth")
        det_data = Data("detections")

    # Directories
    save_dir = Path(increment_path(Path(project_name) / "_".join([task_name]+tags), exist_ok=True))
    save_dir.mkdir(parents=True, exist_ok=True)

    # Load ground truth dataset
    annset_gt = load_dataset(grounds_annfile)

    if add_data_stats:
        plot_data_stats(annset_gt, save_dir / "data_stats.png", log2clearml)

    # Load detections dataset
    annset_pr = load_dataset(detections_annfile)

    nc = len(annset_gt._id_to_label)
    lab2id = {v:k for k, v in annset_gt._id_to_label.items()}
    iouv = np.linspace(0.5, 0.95, 10)
    niou = len(iouv)

    seen = 0
    if add_false_negatives:
        boxes2plot = ObjectSelector(imgs_path)
    confusion_matrix = ConfusionMatrix(nc=nc, log2clearml=log2clearml)
    names = annset_gt._id_to_label
    p, r, mp, mr, map50, map = 0., 0., 0., 0., 0., 0.
    stats: List[Tuple[np.ndarray, np.ndarray, np.ndarray, List[int]]] = []
    imgs4plot = np.zeros((batch_size, 3, imgsz, imgsz))
    gt_labs4plot: List[np.ndarray] = []
    pr_labs4plot: List[np.ndarray] = []
    gt_bboxes4plot: List[np.ndarray] = []
    pr_bboxes4plot: List[np.ndarray] = []
    confs4plot: List[np.ndarray] = []
    ids4plot: List[str] = []
    maps4plot: List[str] = []
    sorted_ids = sorted(list(annset_gt.image_ids))
    for nimg, img_id in enumerate(tqdm(sorted_ids, "process detections")):
        try:
            ann_gt = annset_gt[img_id]
        except KeyError:
            logging.warning(f"Image ID {img_id} not found in ground truth dataset.")
            continue
        if img_id not in annset_pr._annotations.keys():
            logging.warning(f"Image ID {img_id} not found in detections dataset.")
            continue
        ann_pr = annset_pr[img_id]
        bboxes_gt = ann_gt.boxes
        bboxes_pr = ann_pr.boxes

        if add_tide:
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
        tcls = labels[:, 0].tolist() if nl else []
        seen += 1

        if len(pred) == 0:
            if nl:
                stats.append((np.zeros((0, niou)), np.array([]), np.array([]), tcls))
            continue

        # Predictions
        predn = pred.copy()

        # Assign all predictions as incorrect
        correct = np.zeros((pred.shape[0], niou))
        if nl:
            detected = []
            tcls_tensor = labels[:, 0]

            # Target boxes
            tbox = labels[:, 1:5]
            confusion_matrix.process_batch(predn, np.concatenate([labels[:, 0:1], tbox], 1))

            # Per target class
            for cls in np.unique(tcls_tensor):
                ti = np.array((cls == tcls_tensor).nonzero())[0]
                pi = np.array((cls == pred[:, 5]).nonzero())[0]

                # Search for detections
                if pi.shape[0]:
                    # Prediction to target ious
                    iou_matrix = box_iou(predn[pi, :4], tbox[ti])
                    i = iou_matrix.argmax(1)
                    ious = [iou_matrix[irow, icol] for irow, icol in enumerate(i)]
                    # Append detections
                    detected_set = set()
                    for j in (ious >= iouv[0]).nonzero()[0]:
                        d = ti[i[j]]
                        if d.item() not in detected_set:
                            detected_set.add(d.item())
                            detected.append(d)
                            correct[pi[j]] = ious[j] > iouv
                            if len(detected) == nl:
                                break
                    if add_false_negatives:
                        for j in set(ti) - detected_set:
                            boxes2plot.add_fn(img_id, bboxes_gt[j], image_size=ann_gt.image_size)

        # Append statistics (correct, conf, pcls, tcls)
        stats.append((correct, pred[:, 4], pred[:, 5], tcls))

        stats4img = [np.concatenate(x, 0) for x in zip(*stats[-1:])]
        if len(stats4img) and stats4img[0].any():
            p, r, ap, f1, ap_class = ap_per_class(*stats4img, plot=False, save_dir=None, names=names, log2clearml=False)
            ap = [ap[:, 0].mean(), ap[:, :].mean()]
        else:
            ap = [0, 0] if len(tcls) else [1, 1]

        # Plot images
        nplot = int((nimg+1)/batch_size)
        if nplot <= nplots:
            ids4plot.append(img_id)
            maps4plot.append(f"{img_id}")
            img_loader = ImageLoader(imgsz)
            img = img_loader.load_image(imgs_path / img_id).transpose((2, 0, 1))
            gt_bboxes4plot.append(img_loader.correct_bboxes(labels[:, 1:]))
            pr_bboxes4plot.append(img_loader.correct_bboxes(pred[:, :4]))
            gt_labs4plot.append(labels[:, 0])
            pr_labs4plot.append(pred[:, 5])
            confs4plot.append(pred[:, 4])
            imgs4plot[len(ids4plot)-1] = img
            if len(ids4plot) == batch_size:
                f = save_dir / f"batch{nplot}_gtruth.jpg"
                plot_images(imgs4plot, gt_bboxes4plot, gt_labs4plot, None, ids4plot, f, names, log2clearml=log2clearml, save_images=save_images)
                f = save_dir / f"batch{nplot}_pred.jpg"
                plot_images(imgs4plot, pr_bboxes4plot, pr_labs4plot, confs4plot, maps4plot, f, names, log2clearml=log2clearml, save_images=save_images)
                imgs4plot, gt_labs4plot, pr_labs4plot, gt_bboxes4plot, pr_bboxes4plot, confs4plot, ids4plot, maps4plot = np.zeros((batch_size, 3, imgsz, imgsz)), [], [], [], [], [], [], []

    # Plot false negatives
    if add_false_negatives:
        boxes2plot.draw(save_dir / "false_negatives.png", save_images=save_images, log2clearml=log2clearml)

    # Compute statistics
    stats = [np.concatenate(x, 0) for x in zip(*stats)]
    if len(stats) and stats[0].any():
        p, r, ap, f1, ap_class = ap_per_class(*stats, plot=True, save_dir=save_dir, names=names, log2clearml=log2clearml)
        ap50, ap = ap[:, 0], ap.mean(1)
        mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
        nt = np.bincount(stats[3].astype(np.int64), minlength=nc)
    else:
        nt = np.zeros(1)

    # Print results
    columns = ["Class", "Images", "Labels", "P", "R", "mAP@.5%", "mAP@.5:.95%"]
    metrics_table = pd.DataFrame([], columns=columns)
    pf = "%20s" + "%12i" * 2 + "%12.3g" * 4
    logging.info(("%20s" + "%12s" * 2 + "%12s" * 4) % tuple(columns))
    logging.info(pf % ("all", seen, nt.sum(), mp, mr, map50*100, map*100))
    metrics_table = pd.concat(
        [metrics_table, pd.DataFrame([["all", seen, nt.sum(), mp, mr, map50, map]], columns=columns)],
        ignore_index=True)

    # Print results per class
    if nc < 50 and nc > 1 and len(stats):
        for i, c in enumerate(ap_class):
            logging.info(pf % (names[c], seen, nt[c], p[i], r[i], ap50[i], ap[i]))
            metrics_table = pd.concat(
                [metrics_table, pd.DataFrame([[names[c], seen, nt[c], p[i], r[i], ap50[i], ap[i]]], columns=columns)],
                ignore_index=True)
    if log2clearml:
        utils.clearml_task.clearml_logger.report_table("General metrics", "mAPs table", 0, metrics_table)

    metrics_table.to_csv(save_dir / "metrics.csv", index=False)

    # TIDE metrics
    if add_tide:
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
        except Exception as exp:
            logging.error(exp)

    # Plots
    confusion_matrix.plot(save_dir=save_dir, names=list(names.values()))

    # Return results
    logging.info(f"Results saved to {save_dir}")
    return path2log

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(prog="test.py")
    parser.add_argument("--grounds", type=str, help="ground annotations path")
    parser.add_argument("--predicts", type=str, help="predictions path")
    parser.add_argument("--images", type=str, default=None, help="data path")
    parser.add_argument("--weights", type=str, default=None, help="path to model weights to be loaded into clearml")
    parser.add_argument("--batch-size", type=int, default=8, help="size of each image batch")
    parser.add_argument("--save_images", type=bool, default=False, help="save images")
    parser.add_argument("--nplots", type=int, default=5, help="number of plotted batches")
    parser.add_argument("--plot-size", type=int, default=1024, help="size of plotted images")
    parser.add_argument("--project", default="Testing", help="project name in ClearML")
    parser.add_argument("--name", default="example", help="save to project/name")
    parser.add_argument("--tags", type=str, nargs='+', default=[], help="clearml task tags")
    parser.add_argument("--comment", type=str, default=None, help="clearml task comment")    
    parser.add_argument("--data-stats", action="store_true", help="calc and plot statistics of objects sizes")
    parser.add_argument("--tide", action="store_true", help="calc and plot TIDE metric")
    parser.add_argument("--false-negatives", action="store_true", help="add false negatives examples")
    parser.add_argument("--clearml", action="store_true", help="use clearml logging")

    opt = parser.parse_args()
    logging.info(opt)

    path2log = run_test(
        imgs_path=opt.images,
        grounds_annfile=opt.grounds,
        detections_annfile=opt.predicts,
        batch_size=opt.batch_size,
        imgsz=opt.plot_size,
        save_images=opt.save_images,
        nplots=opt.nplots,
        add_data_stats = opt.data_stats,
        add_tide = opt.tide,
        add_false_negatives=opt.false_negatives,
        log2clearml=opt.clearml,
        project_name=opt.project,
        task_name=opt.name,
        task_comment=opt.comment,
        tags=opt.tags
        )
    
    print(path2log)
