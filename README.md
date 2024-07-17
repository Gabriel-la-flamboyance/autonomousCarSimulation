# Project Overview

This project involves simulating an autonomous car using Python. The goal is to train a neural network agent to navigate a car in Udacity's car simulator environment.

## Table of Contents
- Introduction
- Installation
- Detecting lanes
- Training the model
- Results


## Introduction

The objective of this project is to develop an autonomous driving system using reinforcement learning. 
In the first place there is a script that helps to detect and follow road lines and isolates the road region.
In the last place, with the simulator, we train a car to navigate through various tracks. The training is conducted using Convolutional Neural Networks (CNNs) in the Udacity simulator.


## Installation
To get started, you will need the following dependencies:

Python 3.11.9
OpenCV 4.8.1.78
NumPy 1.26.0
Matplotlib 3.8.3

Install the required packages using pip:

    pip install opencv-python numpy matplotlib


## Detecting lanes
Here we first detect the contours, which consists in identifying the sudden changes of intensity in the adjacent pixels. 
![image](https://github.com/user-attachments/assets/1565093b-2b9e-4479-98be-e7d7fbc12770)


Put the image in black is white for ease of processing mages and be fast
![image](https://github.com/user-attachments/assets/9fded69d-521c-4b5e-9d1c-35739634a00f)

Here we remove the noise in the image with a 5x5 Gaussian filter. This also allows a good visualization of our elements.
![image](https://github.com/user-attachments/assets/8813e34d-6e1e-4496-b083-8d460c962d31)


Here we isolate a part of our image, in this case, the road.

With matplotlib, we put the image on a graph, then with the graphic coordinates, we isolate the part that interests us.  
As if by triangulation, we have an area of interest that is isolated. 
![image](https://github.com/user-attachments/assets/b4842cde-050d-4237-9516-bebcf4e9dc43)
![image](https://github.com/user-attachments/assets/cc6830bd-acaa-4dda-af82-33dd67eacaa9)



## Training the model

### The architecture of the implementation
In this step we change the method of working to train a car on the circuit and not just follow a line.
![image](https://github.com/user-attachments/assets/4d78a843-b7cc-44b7-ad50-0c7ea8f8e93f)
![image](https://github.com/user-attachments/assets/b50d6e74-7e80-417e-b47c-318bd9e78839)


### Udacity platform
![image](https://github.com/user-attachments/assets/49227acc-9b29-4be0-af5c-6bf4a4dde370)


#### Autonomous driving simulation
![image](https://github.com/user-attachments/assets/ba4bb04d-ae7e-47dc-8ae4-ff003ee35a6a)


We drive the car for training and the camera on the car takes photos of the track it sees (so we have to drive correctly, otherwise, garbage in, garbage out). 
![image](https://github.com/user-attachments/assets/bbdb696d-acfe-498b-be8e-e031d8259421)



Photos of the car on any side of the track
![image](https://github.com/user-attachments/assets/0337d4ba-c624-4517-bc37-6e779d7c2762)


#### Preprocessing and image augmentation
![image](https://github.com/user-attachments/assets/0691d259-4db6-480b-9d45-1addd909a775)

- Crop the images in the dataset have relevant characteristics in the lower part where the route is visible. 

- Flip (horizontal) the image is returned horizontally (a mirror image of the original image is also stored in the database).  The reason behind this is that the model is trained for the same types of turns on opposite sides.

- Brightness to generalize to weather conditions with a sunny or cloudy day, increasing brightness can be very useful. 


To take into account the distortion effect in the camera while capturing the images, this increase is used because a captured image is not clear every time. Sometimes the camera becomes blurry, but the car must always adapt to this condition and keep the car stable. 


#### Data Filtering

![image](https://github.com/user-attachments/assets/cf15afb5-e1f1-42ff-af20-3df765a8d5f9)

Here, we will filter the images to keep the images where the car is in the middle of the road.   
I consider that tight to the left of the road = -1 , remains in the middle 0 and tight to the right = 1. 
So I filter to only have the data of the car when it is in the middle of the road.  


## Results
After 4 hours of training


https://github.com/user-attachments/assets/1f66dafa-94a1-45fc-836e-0bfff78f8431

Still a lot to do on tris project.
See full project on : https://drive.google.com/file/d/10d5E4V8jDWo2UOTWD-eArOmDa9GD94BB/view?usp=sharing.
