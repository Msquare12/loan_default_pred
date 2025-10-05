import argparse
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from src.models.train import train

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/default.yaml")
    args = ap.parse_args()
    train(args.config)
