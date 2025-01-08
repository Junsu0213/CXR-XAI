# COVID-19 Radiography Classification and Visualization

This project aims to build a deep learning model for classifying and visualizing COVID-19 radiography images. It utilizes the VGG19 model with a Convolutional Block Attention Module (CBAM) to extract features from images and employs Grad-CAM for visualizing model predictions.

## Key Features
- Classification of COVID-19 radiography images
- Visualization of model predictions using Grad-CAM
- Application of various image filtering techniques
- Model evaluation through K-fold cross-validation
- Experiment tracking with Weights & Biases

## Installation and Setup
### Requirements
- Python 3.x
- PyTorch
- torchvision
- wandb
- numpy
- matplotlib
- opencv-python

1. Clone this repository:
    ```bash
    git clone https://github.com/yourusername/COVID19-Radiography-Project.git
    cd COVID19-Radiography-Project
    ```

2. Set up and activate a virtual environment:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

## Usage
1. Run `train_test_main.py` to train the model:
    ```bash
    python train_test_main.py
    ```

2. Run `grad_cam_plot_main.py` to perform Grad-CAM visualization:
    ```bash
    python grad_cam_plot_main.py
    ```

3. Run `kfold_main.py` to perform K-fold cross-validation:
    ```bash
    python kfold_main.py
    ```

## Contributors
- **JUN-SU PARK**
  - Email: junsupark0213@korea.ac.kr

## License
This project is licensed under the MIT License. See the LICENSE file for details. 