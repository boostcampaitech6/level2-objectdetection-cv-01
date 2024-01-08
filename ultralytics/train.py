import argparse
import yaml
import wandb as wb

from ultralytics import YOLO
from ultralytics.utils import checks, yaml_load

from ultralytics.engine.trainer import BaseTrainer
from ultralytics.utils.callbacks.wb import on_pretrain_routine_start

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--cfg", type=str, default="custom.yaml", help="train config file path")
    parser.add_argument("--wandbmode", type=str, default="online", help="wandb logging mode(on: online, off: disabled)")
    parser.add_argument("--data", type=str, default="recycle.yaml", help="explain data_dir and class")
    args = parser.parse_args()

    return args


def main(args):

    cfg = yaml_load(checks.check_yaml(args.cfg))

    model = YOLO(cfg.get('model'))

    try :
        with open(args.data, 'r') as file :
            yaml_data = yaml.safe_load(file)
            data_dir = yaml_data.get('path', None).split("/")[-1]
    except Exception as e :
        raise RuntimeError(f"Error reading YAML file: {e}") from e
    
    if not data_dir :
        raise ValueError("Path not found in YAML file")
    
    wb.init(project=cfg.get('project'),
               name=cfg.get('model')[:-3] + '_' + str(cfg.get('batch')) + '_' + data_dir,
               entity='ai_tech_level2_objectdetection',
               mode=args.wandbmode)

    model.train(cfg=args.cfg, data=args.data)

    wb.finish()

if __name__ == "__main__":
    args = parse_args()

    main(args)