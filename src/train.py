# src/train.py
import os
from ultralytics import YOLO
from roboflow import Roboflow
import wandb  # Optional: for tracking experiments
os.environ["WANDB_API_KEY"] = "d984074aec02b38626a319cbb75d9db22b229d06"



# Initialize Weights & Biases (optional)
wandb.init(project="YOLOv8_Aquarium_Training")

# Define paths and constants
DATASET_PATH = "../dataset/aquarium-data-cots"
EPOCHS = 50
BATCH_SIZE = 16
# DATASET_PATH = "../dataset/aquarium-data-cots"

def download_dataset():
    """Download the dataset from Roboflow if not already present."""
    if not os.path.exists(DATASET_PATH):
        rf = Roboflow(api_key="sIj7hzbYxycpRoHWPGKd")
        project = rf.workspace("model").project("aquarium-data-cots")
        dataset = project.version(1).download("yolov8")
    print("Dataset downloaded and ready!")

def main():
    # Download dataset if not present
    download_dataset()

    # Initialize YOLO model
    model = YOLO("yolov8n.pt")  # Using YOLOv8 Nano model for quicker training

    # Train the model
    results = model.train(
        data=f"{DATASET_PATH}/data.yaml",  # Path to YOLO-compatible dataset YAML file
        epochs=EPOCHS,
        batch=BATCH_SIZE,
        imgsz=640,   # Image size
        name="yolov8_aquarium",  # Model name for saving runs
        augment=True  # Enable augmentation
    )

    # Log metrics to W&B
    wandb.log({"train_loss": results[0], "val_loss": results[1], "mAP": results[2]})
    print("Training completed and metrics logged.")

if __name__ == "__main__":
    main()
