from globox import AnnotationSet
import json
from tqdm import tqdm
from pathlib import Path


def load_dataset(markup_json: str) -> AnnotationSet:
    """
    Load dataset from a json file. Dataset consits of markup file (dict {image_name: properties} and label_names.json (dict class_name: number+1})
    """
    markup = json.load(open(markup_json, "r"))
    
    from globox import AnnotationSet, Annotation, BoundingBox
    annset = AnnotationSet()
    for img_name, ann_dict in tqdm(markup.items(), desc=f"loading {markup_json}"):
        w, h = ann_dict["width"], ann_dict["height"]
        ann = Annotation(img_name, (w, h))
        for i in range(len(ann_dict["bboxes"])):
            bbox, label = ann_dict["bboxes"][i], ann_dict["labels"][i]
            if "scores" in ann_dict.keys():
                conf = ann_dict["scores"][i]
            else:
                conf = None
            bbox = BoundingBox(label=label, 
                            xmin=bbox[0]*w,
                            ymin=bbox[1]*h,
                            xmax=bbox[2]*w,
                            ymax=bbox[3]*h,
                            confidence=conf)
            ann.add(bbox)
        annset.add(ann)

    label_to_id = json.load(open(Path(markup_json).parent / "label_names.json", "r"))
    annset._id_to_label = {v-1:k for k, v in label_to_id.items()}  
    return annset     