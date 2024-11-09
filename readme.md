# YOLOv8 Training Project

This project involves training a YOLOv8 model on an aquarium dataset to detect and classify objects. The project is set up to handle data preparation, model training, evaluation, and logging with Weights and Biases (W&B).

## Project Structure

```plaintext
YOLOv8-training/
│
├── dataset/                    # Directory for datasets (not pushed to GitHub)
├── src/                        # Source code for training and utility functions
│   ├── train.py                # Main training script
│   ├── utils.py                # Utility functions (if any)
│
├── fresh_yolo_env/             # Virtual environment (not pushed to GitHub)
├── new_yolo_env/               # Additional virtual environment (not pushed to GitHub)
├── yolo_env/                   # Another virtual environment (not pushed to GitHub)
│
├── wandb/                      # W&B logs and data (not pushed to GitHub)
├── kaggle.json                 # Kaggle API key (not pushed to GitHub)
├── requirements.txt            # Python dependencies for the project
└── README.md                   # Project documentation (this file)
Requirements
Python 3.12 or later
Dependencies listed in requirements.txt
Setup Instructions
Clone the Repository:

bash
Copy code
git clone https://github.com/your-username/YOLOv8-training.git
cd YOLOv8-training
Create and Activate a Virtual Environment:

bash
Copy code
python3 -m venv fresh_yolo_env
source fresh_yolo_env/bin/activate
Install Dependencies:

Install all required packages listed in requirements.txt.

bash
Copy code
pip install -r requirements.txt
Kaggle API Key:

Place your kaggle.json file in the ~/.kaggle/ directory for accessing the Kaggle API.

bash
Copy code
mkdir -p ~/.kaggle
mv kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
Dataset
Download the aquarium dataset from Kaggle and place it in the dataset/ directory. You can also automate this using the Kaggle API if you have configured your API key.

Aquarium Dataset
Training the Model
To train the model, run the following command:

bash
Copy code
python src/train.py
This script will:

Load the dataset from the specified path.
Initialize and configure the YOLOv8 model.
Train the model using the dataset.
Log metrics and results to Weights and Biases (W&B).
Configurations
You can adjust various training parameters in src/train.py, including:

Learning rate
Batch size
Number of epochs
Data augmentations
Logging and Monitoring
This project uses Weights and Biases (W&B) for experiment tracking. Make sure you have a W&B account and are logged in before running the training script. The script will prompt you to log in if necessary.

To log in:

bash
Copy code
wandb login
Results and Evaluation
After training, you can evaluate the model's performance using metrics like:

Precision
Recall
mAP (mean Average Precision)
The evaluation results will be logged in W&B for detailed analysis.


