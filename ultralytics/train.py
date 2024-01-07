import argparse
import yaml

from ultralytics import YOLO
from ultralytics.utils import checks, yaml_load

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--cfg", type=str, default="custom.yaml", help="train config file path")

    args = parser.parse_args()

    return args

def main(args):

    cfg = yaml_load(checks.check_yaml(args.cfg))

    model = YOLO(cfg.get('model'))

    model.train(cfg=args.cfg, data='recycle.yaml')

if __name__ == "__main__":
    args = parse_args()

    main(args)