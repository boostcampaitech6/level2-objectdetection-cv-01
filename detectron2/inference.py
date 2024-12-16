import os
import copy
import argparse
import pandas as pd

from tqdm import tqdm

from detectron2.data import detection_utils as utils
from detectron2.utils.logger import setup_logger

from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.data.datasets import register_coco_instances
from detectron2.data import build_detection_test_loader

def register_dataset(args):
    try:
        register_coco_instances('coco_trash_test', {}, f'{args.data_dir}/test.json', args.data_dir)
    except AssertionError:
        pass

def setup_config(args):
    cfg = get_cfg()

    cfg.merge_from_file(f'./configs/{args.config_path}/{args.model}.yaml')
    
    cfg.DATASETS.TEST = ('coco_trash_test',)

    cfg.DATALOADER.NUM_WOREKRS = 2

    save_path = os.path.join(args.output_dir, args.model)
    os.makedirs(save_path, exist_ok = True)
    cfg.OUTPUT_DIR = save_path

    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, args.ckpt_name)

    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 10
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.01

    return cfg

def TestMapper(dataset_dict):
    
    dataset_dict = copy.deepcopy(dataset_dict)
    image = utils.read_image(dataset_dict['file_name'], format='BGR')
    
    dataset_dict['image'] = image
    
    return dataset_dict

def inference(predictor):

    test_loader = build_detection_test_loader(cfg, 'coco_trash_test', TestMapper)

    prediction_strings = []
    file_names = []

    for data in tqdm(test_loader):
        
        prediction_string = ''
        
        data = data[0]
        
        outputs = predictor(data['image'])['instances']
        
        targets = outputs.pred_classes.cpu().tolist()
        boxes = [i.cpu().detach().numpy() for i in outputs.pred_boxes]
        scores = outputs.scores.cpu().tolist()
        
        for target, box, score in zip(targets,boxes,scores):
            prediction_string += (str(target) + ' ' + str(score) + ' ' + str(box[0]) + ' ' 
            + str(box[1]) + ' ' + str(box[2]) + ' ' + str(box[3]) + ' ')
        
        prediction_strings.append(prediction_string)
        file_names.append(data['file_name'].replace(args.data_dir,''))

    submission = pd.DataFrame()
    submission['PredictionString'] = prediction_strings
    submission['image_id'] = file_names
    submission.to_csv(os.path.join(cfg.OUTPUT_DIR, f'submission.csv'), index=None)

def parse_args():
    parser = argparse.ArgumentParser(description='Obejct Detection Inference by Detectron2')

    parser.add_argument('--data_dir', type=str, default='/root/dataset/', help='data dir')
    parser.add_argument('--output_dir', type=str, default='./output/')
    parser.add_argument('--config_path', type=str, default='COCO-Detection', help='select config path')
    parser.add_argument('--model', type=str, default='faster_rcnn_R_101_FPN_3x', help='train model name')
    parser.add_argument('--ckpt_name', type=str, default='model_final.pth', help='load model .pth file name')
    args = parser.parse_args()

    return args

if __name__ == '__main__':
    args = parse_args()

    register_dataset(args)

    cfg = setup_config(args)

    predictor = DefaultPredictor(cfg)

    inference(predictor)
