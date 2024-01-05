import os
import copy
import json
import torch
import random
import argparse
import numpy as np

from detectron2.data import detection_utils as utils
from detectron2.utils.logger import setup_logger

from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.evaluation import COCOEvaluator
from detectron2.data import build_detection_train_loader

import detectron2.data.transforms as T

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

def parse_args():
    parser = argparse.ArgumentParser(description='Obejct Detection Train by Detectron2')

    parser.add_argument('--seed', type=int, default=42, help='Fixed Seed')
    parser.add_argument('--data_dir', type=str, default='/root/dataset/', help='Train data dir')
    parser.add_argument('--k_fold', type=int, default=0, help='select k fold number')
    parser.add_argument('--output_dir', type=str, default='./output/')
    parser.add_argument('--config_path', type=str, default='COCO-Detection', help='select config path')
    parser.add_argument('--model', type=str, default='faster_rcnn_R_101_FPN_3x', help='train model name')
    parser.add_argument('--epochs', type=int, default=10, help='train epochs')
    args = parser.parse_args()

    return args

def setup_config(args):
    cfg = get_cfg()

    cfg.merge_from_file(f'./configs/{args.config_path}/{args.model}.yaml')
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(f'{args.config_path}/{args.model}.yaml')

    cfg.DATASETS.TRAIN = ('coco_trash_train',)
    cfg.DATASETS.TEST = ('coco_trash_val',)

    cfg.DATALOADER.NUM_WOREKRS = 2

    cfg.MODEL.MASK_ON = False
    cfg.SOLVER.IMS_PER_BATCH = 4
    
    data = json.load(open(f'{args.data_dir}/train.json'))

    cfg.SOLVER.MAX_ITER = int(len(data['images']) / cfg.SOLVER.IMS_PER_BATCH * args.epochs)
    cfg.SOLVER.BASE_LR = 0.001
    cfg.SOLVER.STEPS = (2000, 4000)
    cfg.SOLVER.GAMMA = 0.005
    cfg.SOLVER.CHECKPOINT_PERIOD = 3000

    save_path = os.path.join(args.output_dir, args.model)
    os.makedirs(save_path, exist_ok = True)
    cfg.OUTPUT_DIR = save_path

    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 10

    cfg.TEST.EVAL_PERIOD = 3000

    return cfg

def register_dataset(args):
    for data in ['train', 'val']:
        try:
            register_coco_instances('coco_trash_' + data, {}, f'{args.data_dir}/{data}_kfold_{args.k_fold}.json', args.data_dir)
        except AssertionError:
            pass

    MetadataCatalog.get("coco_trash_train").set(thing_classes=["General trash", "Paper", "Paper pack", "Metal", 
                                                        "Glass", "Plastic", "Styrofoam", "Plastic bag", "Battery", "Clothing"])

def TrainMapper(dataset_dict):
    dataset_dict = copy.deepcopy(dataset_dict)
    image = utils.read_image(dataset_dict['file_name'], format='BGR')
    
    transform_list = [
        T.Resize((1024, 1024)),
        T.RandomFlip(prob=0.5, horizontal=True, vertical=False),
        T.RandomBrightness(0.8, 1.8),
        T.RandomContrast(0.6, 1.3),
        T.RandomRotation(angle=0.5),
        T.RandomLighting(scale=0.5),
    ]
    
    image, transforms = T.apply_transform_gens(transform_list, image)
    
    dataset_dict['image'] = torch.as_tensor(image.transpose(2,0,1).astype('float32'))
    
    annos = [
        utils.transform_instance_annotations(obj, transforms, image.shape[:2])
        for obj in dataset_dict.pop('annotations')
        if obj.get('iscrowd', 0) == 0
    ]
    
    instances = utils.annotations_to_instances(annos, image.shape[:2])
    dataset_dict['instances'] = utils.filter_empty_instances(instances)
    
    return dataset_dict

class MyTrainer(DefaultTrainer):
    
    @classmethod
    def build_train_loader(cls, cfg, sampler=None):
        return build_detection_train_loader(
        cfg, mapper = TrainMapper, sampler = sampler
        )
    
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            os.makedirs(cfg.OUTPUT_DIR, exist_ok = True)
            output_folder = cfg.OUTPUT_DIR
            
        return COCOEvaluator(dataset_name, cfg, False, output_folder)

if __name__ == '__main__':
    args = parse_args()
    seed_everything(args.seed)

    register_dataset(args)
    
    cfg = setup_config(args)

    trainer = MyTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()

# logger = setup_logger()