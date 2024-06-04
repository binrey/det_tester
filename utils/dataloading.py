from globox import AnnotationSet
import json
from tqdm import tqdm

class MyAnnotationSet(AnnotationSet):
    """
    MyAnnotationSet is a subclass of AnnotationSet.
    """
    pass

    def from_markup(markup_json, labels):
        markup = json.load(open(markup_json, "r"))
        
        from globox import AnnotationSet, Annotation, BoundingBox
        annset = AnnotationSet()
        for img_name, ann_dict in tqdm(markup.items()):
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

        annset._id_to_label = {i:lab for i, lab in enumerate(labels)}  
        return annset      