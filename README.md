# DAM-CGNet: Semantic Segmentation-Based Approach for Valley Bottom Extraction from Digital Elevation Models (DEMs)
This repository provides the code for DAM-CGNet, a semantic segmentation model designed for accurate valley-bottom extraction from DEMs. The model  improves valley-bottom extraction by addressing the limitations of conventional threshold-based methods.

# Requirements

- CUDA-enabled GPU
- Python 3.8.2
- PyTorch 1.7.2
- GDAL  3.4.1
- torchvision  0.11.2
- opencv-python 4.6.0.66

# Data
The training data for the model were sourced from the Ohio Statewide Imagery Program (OSIP).

 https://gis1.oit.ohio.gov/geodatadownload/ 



## Code Structure

- Run `python train.py` to train DAM-CGNet.

- Run `python test.py` to predict on the trained DAM-CGNet.
- The sample folder holds the test data.
- Download the trained DAM-CGNet and put it in the "weights" folder to predict samples.

### Download trained DAM-CGNet
- The trained model is available for [Google Drive](https://drive.google.com/file/d/1ZGo6icASXNpLeoW1rqWfwlKi-PUkitWf/view?usp=drive_link). To ensure proper access and usage, please follow these steps:

  Click on the Google Drive link.

  Send a request for access by clicking the "Request" button.
  Once your access is granted, you can download the model file.
  Thank you for your understanding, and please feel free to reach out if you encounter any issues.
