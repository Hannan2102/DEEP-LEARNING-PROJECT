# Single Image Depth Estimation using UNet

This project implements a deep learning pipeline to estimate depth from a single RGB image using a UNet-based model. It was trained on the NYU Depth V2 dataset and is built entirely using PyTorch.


## Overview

Given a single RGB image, the model predicts a dense depth map, simulating how far each pixel is from the camera. This is useful in fields like robotics, AR/VR, and 3D reconstruction.

We use:
- A custom PyTorch `Dataset` class
- UNet architecture with configurable encoders
- Scale-invariant loss for better learning of relative depth
- Evaluation metrics and visualizations to assess performance


## Folder Structure

DEEP-LEARNING-PROJECT/
├── hannan_final_dl.ipynb # Main notebook with all training and evaluation steps
├── README.md # Project documentation
├── /nyu_data2/nyu2_train/ # Training RGB and Depth images
├── /nyu_data2/nyu2_test/ # Testing RGB and Depth images



##  Model Architecture

We use a standard **UNet** architecture:

- Encoder: configurable (default uses pretrained ResNet-like features)
- Decoder: upsampling path with skip connections
- Loss: **Scale-Invariant Loss** to focus on relative depth, not just absolute pixel values



##  Evaluation Metrics

We evaluate depth predictions using:
- Absolute Relative Error (AbsRel)
- RMSE (Root Mean Squared Error)
- Threshold Accuracy δ<1.25, δ<1.25², δ<1.25³


##  Sample Output

Predictions are visualized side-by-side:

[ RGB Image | Ground Truth Depth | Predicted Depth ]


Each output is saved as an image using matplotlib.

---

## How to Run

1. Clone the repo:
   ```bash
   git clone https://github.com/Hannan2102/DEEP-LEARNING-PROJECT.git
   cd DEEP-LEARNING-PROJECT

2. Prepare the dataset:
    Download NYU Depth V2 and place it in:
    /Users/abdulhannan/Desktop/nyu_data2/
    ├── nyu2_train/
    └── nyu2_test/

3. Open and run hannan_final_dl.ipynb in Jupyter or Colab
   
4. The model will:
    Train on ~6000 samples
    Validate and test on separate splits
    Visualize predictions
    Save predicted depth images and the trained model

Libraries Used

Python
PyTorch
NumPy
Matplotlib
PIL
Torchvision
