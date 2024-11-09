# YOLOv8 Aquarium Detection Project

## Project Overview
This project trains a YOLOv8 model on the [Aquarium Dataset](https://www.kaggle.com/datasets/slavkoprytula/aquarium-data-cots) to detect and classify marine objects. The goal is to gain insights into object detection using YOLOv8 and explore various training configurations and analysis.

---

## Dataset
- **Source**: [Kaggle Aquarium Data](https://www.kaggle.com/datasets/slavkoprytula/aquarium-data-cots)
- **Format**: YOLO format (converted via Roboflow)
- **Structure**:
  - `images/`: Contains all image files.
  - `labels/`: Contains label files with bounding box coordinates.
  - `data.yaml`: Config file specifying dataset paths and class names.

---

## Setup and Installation

### Prerequisites
- Python 3.7 or higher
- VS Code with Python extension

### Install Dependencies
1. Clone this repository:
   ```bash
   git clone <repository_url>
   cd YOLOv8_Aquarium_Project
