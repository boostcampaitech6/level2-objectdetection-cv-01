import argparse

from ultralytics import YOLO

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--cfg", type=str, default="custom.yaml", help="train config file path")
    parser.add_argument("--weight", type=str, default="yolov8x.pt", help="yolov8 model weight(.pt) or path(outputs/~/best.pt)")

    args = parser.parse_args()

    return args

def main(args):

    model = YOLO(args.weight)

    model.train(cfg=args.cfg, data='recycle.yaml')

if __name__ == "__main__":
    args = parse_args()
    
    main(args)