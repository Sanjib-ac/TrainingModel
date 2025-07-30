import torch
import argparse
import os
import sys
import yaml
from ultralytics import YOLO

# print("CUDA available:", torch.cuda.is_available())
# print("CUDA version:", torch.version.cuda)
# print("Current working directory:", os.getcwd())
# print("Python executable:", sys.executable)
# print("Arguments received:", sys.argv)


def is_multiprocessing_worker():
    """Check if this is a multiprocessing worker process"""
    return '--multiprocessing-fork' in sys.argv or 'parent_pid=' in ' '.join(sys.argv)


def parse_args():
    if is_multiprocessing_worker():
        # print("Detected multiprocessing worker - skipping argument parsing")
        return None

    parser = argparse.ArgumentParser(description="Train a model from the command line with all "
                                                 "available parameters")
    parser.add_argument("--task", type=str, choices=["detect", "segment", "classify"],
                        default="detect", help="Task type: detect (default), segment, or classify")

    # Core
    parser.add_argument("--model", type=str, required=True,
                        help="weights or checkpoint(e.g., model.pt, or .yaml)")
    parser.add_argument("--data", type=str, required=True,
                        help="Dataset config YAML (train/val paths + class names)")
    parser.add_argument("--pretrained", type=str, default=None,
                        help="Optional pretrained weights (.pt)")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Total training epochs")
    parser.add_argument("--batch", type=int, default=4,
                        help="Batch size per GPU")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="Input image size (pixels)")
    parser.add_argument("--device", type=str, default="0",
                        help="CUDA device(s), e.g., '0', '0,1' or 'cpu'")
    parser.add_argument("--val", action="store_true", default=False, help="Enables validation during training")

    # Saving & resuming
    parser.add_argument("--project", type=str, default="train",
                        help="Root directory for saving results")
    parser.add_argument("--name", type=str, default="ADSTraining",
                        help="Trained model name (folder under project)")
    parser.add_argument("--exist_ok", action="store_true", default=False, help="If True, allows overwriting of an "
                                                                               "existing project/name directory. ")

    #
    parser.add_argument("--workers", type=int, default=8, help="Number of data loading workers")
    parser.add_argument("--rect", action="store_true", help="Rectangular training")
    parser.add_argument("--single_cls", action="store_true", help="Treats all classes in as a single class")
    parser.add_argument("--multi_scale", action="store_true", help="increasing/decreasing imgsz by up to a factor of "
                                                                   "0.5")

    # Optimizer & LR scheduler
    parser.add_argument("--optimizer", type=str, default="SGD",
                        help="Optimizer (SGD, Adam, etc.)")

    # Miscellaneous
    parser.add_argument("--patience", type=int, default=50, help="Early stopping patience (no. of epochs)")
    parser.add_argument("--verbose", action="store_true", help="Verbose output during training")
    parser.add_argument("--shear", type=float, default=0.0, help="Shears the image by a specified degree, mimicking "
                                                                 "the effect of objects being viewed from different "
                                                                 "angles.")
    parser.add_argument("--degrees", type=float, default=0.0, help="Rotates the image randomly within the specified "
                                                                   "degree range, improving the model's ability to "
                                                                   "recognize objects at various orientations.")
    parser.add_argument("--bgr", type=float, default=0.0)
    parser.add_argument("--mixup", type=float, default=0.0, help="Blends two images and their labels, creating a "
                                                                 "composite image")

    return parser.parse_args()


def main():
    print("CUDA available:", torch.cuda.is_available())
    print("CUDA version:", torch.version.cuda)
    print("CWD:", os.getcwd())
    print("Python exe:", sys.executable)
    try:
        args = parse_args()
        if args is None:
            return
        # Verify files exist
        print(f"Checking model file: {args.model}")
        if not os.path.exists(args.model):
            print(f"ERROR: Model file not found: {args.model}")
            print(f"Full path: {os.path.abspath(args.model)}")
            return
        else:
            print(f"Model file exists: {args.model}")
            print(f"File size: {os.path.getsize(args.model)} bytes")

        print(f"Checking data file: {args.data}")
        if not os.path.exists(args.data):
            print(f"ERROR: Data file not found: {args.data}")
            print(f"Full path: {os.path.abspath(args.data)}")
            return
        else:
            print(f"Data file exists: {args.data}")
            print(f"File size: {os.path.getsize(args.data)} bytes")

            # Try to read the YAML file
            try:
                with open(args.data, 'r') as f:
                    data_config = yaml.safe_load(f)
                print(f"YAML file is valid, contains: {list(data_config.keys())}")
            except Exception as e:
                print(f"ERROR: Could not read YAML file: {e}")
                return

        # Map args to train() kwargs
        train_kwargs = {
            "task": args.task,
            "model": args.model,
            "data": args.data,
            "epochs": args.epochs,
            "batch": args.batch,
            "imgsz": args.imgsz,
            "device": args.device,
            "pretrained": args.pretrained,
            "project": args.project,
            "name": args.name,
            "workers": args.workers,
            "rect": args.rect,
            "single_cls": args.single_cls,
            "multi_scale": args.multi_scale,
            "mixup": args.mixup,
            "optimizer": args.optimizer,
            "patience": args.patience,
            "verbose": args.verbose,
        }

        print("DEBUG: train_kwargs =", train_kwargs)

        print(f"Loading YOLO model from {args.model}")
        model = YOLO(args.model)
        print(f"Loaded YOLO model successfully")
        print(f"Model info: {model.info()}")

        try:
            print(f"Starting {args.task} training...")
            if args.task == "detect":
                results = model.train(**train_kwargs)
            elif args.task == "segment":
                results = model.train(**train_kwargs)
            elif args.task == "classify":
                results = model.train(**train_kwargs)
            else:
                raise ValueError(f"Unknown task: {args.task!r}")

            print("✓ Training completed successfully!")
            print(f"Results: {results}")

        except Exception as e:
            print(f"ERROR during training: {e}")
            import traceback
            traceback.print_exc()
            raise

    except Exception as e:
        print(f"ERROR in main: {e}")
        import traceback
        traceback.print_exc()
        input("Press Enter to exit...")  # Keep console open to see error


if __name__ == "__main__":
    # Prevent multiprocessing issues with PyInstaller
    if hasattr(sys, 'frozen'):
        import multiprocessing

        multiprocessing.freeze_support()
    main()
