import argparse
import os

import pandas as pd
import torch
from pycocotools.coco import COCO
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from ultralytics import YOLO

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default= '/data/ephemeral/home/level2-objectdetection-cv-01/ultralytics/YOLOv8/train/weights/best.pt',help="Specify the file path of the model for training")
    parser.add_argument("--dataset_path", type=str, default="/data/ephemeral/home/dataset_final/images", help="Set the file path where the training dataset images are stored")
    parser.add_argument("--json_path", type=str, default="/data/ephemeral/home/dataset_final/test.json", help="Define the file path for the test JSON file")
    args = parser.parse_args()
    return args

class CustomDataset(Dataset):
    def __init__(self, annotation, data_dir):
        super().__init__()
        self.data_dir = data_dir
        self.coco = COCO(annotation)

    def __getitem__(self, index: int):
        image_id = self.coco.getImgIds(imgIds=index)
        image_info = self.coco.loadImgs(image_id)[0]
        image = os.path.join(self.data_dir, image_info["file_name"])

        return image

    def __len__(self) -> int:
        return len(self.coco.getImgIds())

def get_bboxes(annotation, data_loader, model):
    prediction_strings = []
    file_names = []

    COCO(annotation)
    for _, image_batch in enumerate(tqdm(data_loader)):

        with torch.no_grad():
            predictions = model(image_batch)
        batch_size = len(image_batch)
        for idx in range(batch_size):
            file_name = os.path.join("test", os.path.basename(image_batch[idx]))
            pred_info = predictions[idx].boxes
            classes = pred_info.cls.tolist()
            bboxes = pred_info.xyxy.tolist()
            conf = pred_info.conf.tolist()

            prediction_string = ""
            for pred_class, conf, box in zip(classes, conf, bboxes):
                prediction_string += \
                    (str(int(pred_class)) + " " + str(conf) + " " + 
                     str(box[0]) + " " + str(box[1]) + " " + str(box[2]) + " " + str(box[3]) + " ")
                
            prediction_strings.append(prediction_string)
            file_names.append(file_name)

    return prediction_strings, file_names

def inference(args):
    model = YOLO(args.model)

    # annotation 경로
    annotation = args.json_path # ex. "/data/ephemeral/home/dataset_yolo/test.json"
    data_dir = args.dataset_path # ex. "/data/ephemeral/home/dataset_yolo/images"

    # 데이터셋 로드
    test_dataset = CustomDataset(annotation, data_dir)
    test_data_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=4)

    # 예측 및 submission 파일 생성
    prediction_strings, file_names = get_bboxes(annotation, test_data_loader, model)
    submission = pd.DataFrame()
    submission["PredictionString"] = prediction_strings
    submission["image_id"] = file_names
    
    submission.to_csv("./yolo_submission.csv", index=None)
    print(submission.head())


if __name__ == "__main__":
    args = parse_args()

    inference(args)
