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
import numpy as np

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


def visualize_predictions(dataset_name: str, cfg_path: str, model_path: str):
    dataset_dicts = DatasetCatalog.get(dataset_name)
    model_root = os.path.dirname(model_path)
    cfg = LazyConfig.load(cfg_path)
    cfg.train.init_checkpoint = model_path
    model = instantiate(cfg.model)
    model.to(cfg.train.device)
    checkpointer = DetectionCheckpointer(model)
    checkpointer.load(cfg.train.init_checkpoint)

    model.eval()

    demo = VisualizationDemo(
        model=model,
        metadata_dataset=dataset_name
    )

    image_dir = os.path.join(model_root, "predictions")
    os.makedirs(image_dir, exist_ok=True)

    bbox_predictions = []

    for entry in dataset_dicts:
        path = entry["file_name"]
        img = np.expand_dims(cv2.imread(path, cv2.IMREAD_GRAYSCALE), -1)
        predictions, visualized_output = demo.run_on_image(img, 0.1)
        cv2.imwrite(os.path.join(image_dir, path.split("/")[-1]), visualized_output.get_image()[:, :, ::-1])
        pred_bboxes = predictions["instances"]._fields["pred_boxes"].tensor.tolist()
        pred_scores = predictions["instances"]._fields["scores"].tolist()
        pred_classes = predictions["instances"]._fields["pred_classes"].tolist()
        parsed_predictions = []
        for i in range(len(pred_bboxes)):
            parsed_prediction = pred_bboxes[i]
            parsed_prediction.append(pred_scores[i])
            parsed_prediction.append(pred_classes[i])
            parsed_predictions.append(parsed_prediction)
        bbox_predictions.append({
            "image": os.path.join(image_dir, path),
            "predictions": parsed_predictions
        })

    with open(os.path.join(image_dir, "predictions.json"), "w") as f:
        json.dump(bbox_predictions, f)
