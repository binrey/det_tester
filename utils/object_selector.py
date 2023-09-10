from globox import BoundingBox
import cv2
import numpy as np
from matplotlib import pyplot as plt
from random import shuffle
from tqdm import tqdm


class ObjectSelector:
    def __init__(self, imgs_path, max_count=100):
        self.fn = {}
        self.max_count = max_count
        self.imgs_path = imgs_path
    
    def add_fn(self, img_id:str, box:BoundingBox, image_size=None):
        label = box.label
        l, t, r, b = map(int, box.ltrb)
        if image_size is None or (min(l, r) < image_size[0] and min(t, b) < image_size[1]):
            item2add = {"img_id": img_id, "ltrb": (l, t, r, b)}
            if label not in self.fn.keys():
                self.fn[label] = [item2add]
            elif len(self.fn[label]) < self.max_count:
                self.fn[label].append(item2add)
        else:
            print(f"!!! Wrong box in {img_id} for label {label}")
            
    @staticmethod
    def get_optimal_font_scale(text, size):
        for scale in reversed(range(0, 60, 1)):
            textSize = cv2.getTextSize(text, fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=scale/10, thickness=1)
            new_height = textSize[0][1]
            if (new_height <= size):
                return scale/10, textSize[0][0], textSize[0][1]
        return 1

    def plot_fn(self, path2save, batch_size=5, img_size=(120, 120)):
        self.draw(self.fn, path2save, batch_size, img_size)

    def draw(self, examples, path2save, ncols=5, img_size=(120, 120), pad=10):
        irow = 0
        for lab in tqdm(examples.keys(), "process false negatives"):
            if len(examples[lab]) == 0:
                continue
            shuffle(examples[lab])
            for c in range(ncols):
                if c < len(examples[lab]):
                    impath = self.imgs_path / examples[lab][c]["img_id"]
                    l, t, r, b = examples[lab][c]["ltrb"]
                    img = cv2.imread(str(impath))
                    img = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=(0, 0, 0))
                    img_crop = img[t:b+2*pad, l:r+2*pad]
                    cv2.rectangle(img_crop, (pad-1, pad-1), (img_crop.shape[1]-pad+1, img_crop.shape[0]-pad+1), (255, 255, 0))
                    # cv2.imwrite(f"crops/{lab}{c}.png", img_crop)
                    if img_crop.ndim == 2:
                        img_crop = np.dstack([img_crop]*3)
                    img_crop = cv2.cvtColor(cv2.resize(img_crop, img_size), cv2.COLOR_BGR2RGB)     
                else:
                    img_crop = np.ones((img_size[0], img_size[1], 3), dtype=np.uint8)*255
                if c == 0:
                    row = img_crop
                else:
                    row = np.hstack([row, img_crop])  
                    if c == 4:
                        ts, tw, th = self.get_optimal_font_scale(lab, img_size[1]*0.3)
                        tw, th = tw+4, th+4
                        # cv2.rectangle(row, (0, 0), (tw+3, th+3), (255, 255, 255), -1)
                        white_rect = np.ones((th, tw, 3), dtype=np.uint8) * 255
                        row[:th, :tw] = cv2.addWeighted(row[:th, :tw], 0.5, white_rect, 0.5, 1.0)
                        row = cv2.putText(row, f"{lab}", (0, th), cv2.FONT_HERSHEY_DUPLEX, ts, (0, 0, 0), 2)
                        # row = cv2.putText(row, f"{lab}:{cc[lab]}", (0, th), cv2.FONT_HERSHEY_DUPLEX, ts, (0, 0, 0))
            if irow == 0:
                wall = row
            else:
                wall = np.vstack([wall, row])
            irow += 1


        fig, ax = plt.subplots(figsize=(ncols*2, 2*irow))
        plt.imshow(wall)
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(str(path2save))