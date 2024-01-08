import argparse
import yaml
import wandb as wb

from ultralytics import YOLO
from ultralytics.utils import checks, yaml_load

from ultralytics.engine import trainer

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--cfg", type=str, default="custom.yaml", help="train config file path")

    args = parser.parse_args()

    return args

def on_pretrain_routine_start(trainer) :
    try :
        with open(trainer.args.data, 'r') as file :
            yaml_data = yaml.safe_load(file)
            data_dir = yaml_data.get('path', None)
    except Exception as e :
        raise RuntimeError(f"Error reading YAML file: {e}") from e
    
    if not data_dir :
        raise ValueError("Path not found in YAML file")
    
    wb.run or wb.init(project=trainer.args.project or 'YOLOv8',
                      name=f'{trainer.args.model}_{trainer.args.batch}_{data_dir.split("/")[-2:]}',
                      config=vars(trainer.args),
                      entity='ai_tech_level2_objectdetection')


def main(args):

    cfg = yaml_load(checks.check_yaml(args.cfg))

    model = YOLO(cfg.get('model'))

    model.train(cfg=args.cfg, data='recycle.yaml')

if __name__ == "__main__":
    args = parse_args()

    main(args)




