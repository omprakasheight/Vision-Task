import os
from dotenv import load_dotenv
from ultralytics import YOLO
from roboflow import Roboflow
import wandb  # Optional: for tracking experiments

# Load environment variables from .env file
load_dotenv()

# Retrieve API keys from environment variables
roboflow_api_key = os.getenv("ROBOFLOW_API_KEY")
wandb_api_key = os.getenv("WANDB_API_KEY")

# Ensure API keys are available
if not roboflow_api_key or not wandb_api_key:
    raise ValueError("API keys are missing. Please check your .env file.")

# Set the WANDB API key for logging experiments
os.environ["WANDB_API_KEY"] = wandb_api_key

# Initialize Weights & Biases (optional)
wandb.init(project="YOLOv8_Aquarium_Training")

# Define paths and constants
DATASET_PATH = "../dataset/aquarium-data-cots"
EPOCHS = 50
BATCH_SIZE = 16

def download_dataset():
    """Download the dataset from Roboflow if not already present."""
    if not os.path.exists(DATASET_PATH):
        rf = Roboflow(api_key=roboflow_api_key)
        project = rf.workspace("model").project("aquarium-data-cots")
        
        # List all versions available in the project
        available_versions = project.versions()
        print("Available versions:", available_versions)
        
        # Attempt to download the dataset from an available version
        try:
            dataset = project.version("1").download("yolov8")
        except RuntimeError:
            try:
                dataset = project.version("v1").download("yolov8")
            except RuntimeError:
                # Attempt using integer for version number
                try:
                    dataset = project.version(1).download("yolov8")
                except RuntimeError:
                    print("Error: None of the specified versions were found. Please check the available versions.")
                    return

        print("Dataset downloaded and ready!")

def main():
    # Download dataset if not present
    download_dataset()

    # Initialize YOLO model
    model = YOLO("yolov8n.pt")  # Using YOLOv8 Nano model for quicker training

    # Train the model
    results = model.train(
    data="dataset/aquarium-data-cots/aquarium_pretrain/data.yaml",  # Path to YOLO-compatible dataset YAML file
    epochs=EPOCHS,
    batch=BATCH_SIZE,
    imgsz=640,  # Image size
    name="yolov8_aquarium",  # Model name for saving runs
    augment=True  # Enable augmentation
)


    # Log metrics to W&B
#     wandb.log({"train_loss": results[0], "val_loss": results[1], "mAP": results[2]})
#     print("Training completed and metrics logged.")

# if __name__ == "__main__":
#     main()
def main():
    # Download dataset if not present
    download_dataset()

    # Initialize YOLO model
    model = YOLO("yolov8n.pt")  # Using YOLOv8 Nano model for quicker training

    # Train the model
    # 
    results = model.train(
    data="dataset/aquarium-data-cots/aquarium_pretrain/data.yaml",
    epochs=50,
    batch=BATCH_SIZE,  # Smaller batch size for faster testing
    imgsz=640,
    name="yolov8_aquarium",
    augment=True
)


    # Access and log metrics
    metrics = results.results_dict

    print("Available metrics:", metrics)  # Print available metrics for verification

    # Log specific metrics to W&B (adjust keys as needed based on output)
    wandb.log({
        "train_loss": metrics.get("box_loss"),
        "val_loss": metrics.get("cls_loss"),
        "mAP50": metrics.get("mAP50"),
        "mAP50-95": metrics.get("mAP50-95")
    })
    print("Training completed and metrics logged.")

if __name__ == "__main__":
    main()
