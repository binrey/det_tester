from globox import AnnotationSet, Annotation, BoundingBox
from pathlib import Path
from PIL import Image


class AnnotationLoader:
    def __init__(self, ann_file, imgs_path):
        self.annset = AnnotationSet.from_coco(ann_file)
        self.imgs_path = Path(imgs_path)
        self.image_ids = list(self.annset.image_ids)

    def __len__(self):
        return len(self.annset.image_ids)
    
    def __getitem__(self, index):
        img_id = self.image_ids[index]
        ann_item = self.annset[img_id]
        img_path = self.imgs_path / img_id
        return img_path


if __name__ == "__main__":
    aloader = AnnotationLoader("test_annotation.json", "../assets/nlmk11/images/test")
    print(len(aloader))
    print(aloader.annset.show_stats())
    print(aloader[0])