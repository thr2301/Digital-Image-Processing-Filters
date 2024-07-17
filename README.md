# Digital-Image-Processing-Filters
This is a project for the university course Digital Image Processing and was graded 10/10
# Image Processing Project

This project contains Python scripts for various image processing tasks implemented using NumPy and matplotlib.

## Table of Contents
- [Introduction](#introduction)
- [Functionality](#functionality)
  - [Image Patches](#image-patches)
  - [Convolution](#convolution)
  - [Edge Detection](#edge-detection)
  - [Sobel Operator](#sobel-operator)
  - [Laplacian of Gaussian (LoG) Filter](#laplacian-of-gaussian-log-filter)

## Introduction

This repository contains Python scripts for performing several fundamental image processing tasks. The scripts utilize NumPy for numerical operations and matplotlib for visualizations.The common.py was provided by the professor and includes some helper functions.

## Functionality

### Original Image
![grace_hopper](https://github.com/user-attachments/assets/87191cf3-e398-4c80-9dd8-fac93bda0de2)

### Image Patches

Extracts patches from an input image and normalizes them based on their mean and standard deviation.
![3_patches](https://github.com/user-attachments/assets/a01d9743-200b-4665-9c76-232fe6903a64)

### Convolution

Implements convolution with zero-padding using custom functions for correctness verification.
![q2_gaussian](https://github.com/user-attachments/assets/9c8e5281-827d-41a5-ac9f-1ff652db5e99)

### Edge Detection

Performs edge detection using both Sobel operators and Laplacian of Gaussian (LoG) filters.
![edge_detection_comparison](https://github.com/user-attachments/assets/4dabf57c-baed-41f3-b508-e7372052b37c)

### Sobel Operator

Applies Sobel operators to compute gradients and gradient magnitudes of an input image.
![q2_Gy](https://github.com/user-attachments/assets/4ac49ad1-d2a3-4c3a-87f0-fcbf8b511dab)
![q2_grad_magnitude](https://github.com/user-attachments/assets/75b7f6b8-397f-4d2b-8976-8213408d4ed1)
![q2_Gx](https://github.com/user-attachments/assets/8c4c2f6e-f657-429e-b9c8-8a4e82cee410)

### Laplacian of Gaussian (LoG) Filter

Applies LoG filters to detect edges and points of interest in an image.
![q1_LoG2](https://github.com/user-attachments/assets/6695647f-1a7e-42a5-a105-402a6c6869f5)
![q1_LoG1](https://github.com/user-attachments/assets/a6ed2c1d-d052-4098-881e-dca97b53f5a1)


Each function is implemented in a modular way within the `main.py` script, demonstrating various image processing techniques.
## Usage

To run the game use:

```
python filters.py
```
You must the image in the same folder with the filters.py
