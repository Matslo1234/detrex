import os
import json
import cv2
import random
from PIL import Image
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import BoxMode
# Some basic setup:

# import some common libraries
import os, json, cv2, random

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.engine import DefaultTrainer
from detectron2.utils.visualizer import ColorMode
from detrex.config import get_config
from detectron2.config import LazyConfig, instantiate
from demo.predictors import VisualizationDemo
from detectron2.checkpoint import DetectionCheckpointer




def create_label_json(img_dir):
    first_img_path = next(os.walk(img_dir))[2][0]
    first_img = Image.open(os.path.join(img_dir, first_img_path)).convert('RGB')
    width, height = first_img.size
    label_dir = os.path.join(img_dir, "..", "labels")

    labels = []
    for idx, label in enumerate(sorted(os.listdir(label_dir))):
        with open(os.path.join(label_dir, label), "r") as f:
            label_dict = {
                "file_name": os.path.join(img_dir, label.replace(".txt", ".PNG")),
                "image_id": idx,
                "height": height,
                "width": width,
                "annotations": []
            }

            for text in f.readlines():
                text = text.strip()
                split = text.split(" ")
                class_id = int(split[0])
                center_x = int(float(split[1]) * width)
                center_y = int(float(split[2]) * height)
                label_width = int(float(split[3]) * width)
                label_height = int(float(split[4]) * height)

                label_dict["annotations"].append({
                    "bbox": [center_x - label_width / 2, center_y - label_height / 2, label_width, label_height],
                    "bbox_mode": BoxMode.XYWH_ABS,
                    "category_id": class_id
                })
            labels.append(label_dict)

    with open(os.path.join(img_dir, "labels.json"), "w") as f:
        json.dump(labels, f)
    return labels


def get_uxo_dicts(img_dir):
    labels_json = os.path.join(img_dir, "labels.json")
    if not os.path.isfile(labels_json):
        labels = create_label_json(img_dir)

    else:
        with open(labels_json, "r") as f:
            labels = json.load(f)

    return labels

def register_uxo_11(img_dir, dataset_prefix, classes):
    for img_type in ["train", "valid", "test"]:
        dataset_name = dataset_prefix + img_type
        DatasetCatalog.register(dataset_name,
                                lambda x=img_type: get_uxo_dicts(os.path.join(img_dir, x, "images")))
        MetadataCatalog.get(dataset_name).set(thing_classes=classes)


register_uxo_11(os.path.join(os.getenv("DETECTRON2_DATASETS", "~/datasets"), "Novi11"), "uxo_11_", ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10"])
register_uxo_11(os.path.join(os.getenv("DETECTRON2_DATASETS", "~/datasets"), "Novi11Resized"), "uxo_11_resized_", ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10"])
register_uxo_11(os.path.join(os.getenv("DETECTRON2_DATASETS", "~/datasets"), "Novi11Fair"), "uxo_11_fair_", ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10"])
register_uxo_11(os.path.join(os.getenv("DETECTRON2_DATASETS", "~/datasets"), "Novi11Fair1Class"), "uxo_11_fair_1class_", ["0"])
register_uxo_11(os.path.join(os.getenv("DETECTRON2_DATASETS", "~/datasets"), "Novi11CroppedFair1Class"), "uxo_11_fair_cropped_1class_", ["0"])
register_uxo_11(os.path.join(os.getenv("DETECTRON2_DATASETS", "~/datasets"), "Novi11OneChannel"), "uxo_11_1channel_", ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10"])
register_uxo_11(os.path.join(os.getenv("DETECTRON2_DATASETS", "~/datasets"), "Novi11OneChannelFair1Class"), "uxo_11_1channel_fair_", ["0"])
register_uxo_11(os.path.join(os.getenv("DETECTRON2_DATASETS", "~/datasets"), "Novi11Fair2561Class1Channel"), "uxo_11_cropped_256_1class_", ["0"])
