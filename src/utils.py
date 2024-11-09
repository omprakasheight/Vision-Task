# src/utils.py
import os
import yaml

def check_dataset_format(path):
    """
    Checks if the dataset format is correct for YOLO training.
    Ensures 'images' and 'labels' directories exist and 'data.yaml' is present.
    """
    required_files = ["images", "labels", "data.yaml"]
    missing_files = [file for file in required_files if not os.path.exists(os.path.join(path, file))]
    
    if missing_files:
        print(f"Missing required files for YOLOv8: {missing_files}")
        return False
    print("Dataset format is correct.")
    return True

def load_yaml(file_path):
    """
    Loads a YAML file and returns its contents as a dictionary.
    Useful for loading YOLO dataset config.
    """
    with open(file_path, 'r') as file:
        data = yaml.safe_load(file)
    return data

def log_metrics(metrics):
    """
    Logs metrics to the console or a tracking service.
    """
    print("Training Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value}")

def apply_augmentation_settings(model, augment=True):
    """
    Apply augmentation settings to the model during training if enabled.
    """
    if augment:
        print("Augmentations enabled: Flipping, scaling, color jitter, etc.")
        model.augment = True  # Adjust model config for augmentations
    else:
        print("No augmentations applied.")
        model.augment = False
