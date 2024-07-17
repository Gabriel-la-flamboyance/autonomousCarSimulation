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


Put the image in black is white for ease of processing mages and be fast

Here we remove the noise in the image with a 5x5 Gaussian filter. This also allows a good visualization of our elements.


Here we isolate a part of our image, in this case, the road.

With matplotlib, we put the image on a graph, then with the graphic coordinates, we isolate the part that interests us.  
As if by triangulation, we have an area of interest that is isolated. 
)


## Training the model

### The architecture of the implementation
In this step we change the method of working to train a car on the circuit and not just follow a line.


### Udacity platform


#### Autonomous driving simulation


We drive the car for training and the camera on the car takes photos of the track it sees (so we have to drive correctly, otherwise, garbage in, garbage out). 


Photos of the car on any side of the track


#### Preprocessing and image augmentation


- Crop the images in the dataset have relevant characteristics in the lower part where the route is visible. 

- Flip (horizontal) the image is returned horizontally (a mirror image of the original image is also stored in the database).  The reason behind this is that the model is trained for the same types of turns on opposite sides.

- Brightness to generalize to weather conditions with a sunny or cloudy day, increasing brightness can be very useful. 


To take into account the distortion effect in the camera while capturing the images, this increase is used because a captured image is not clear every time. Sometimes the camera becomes blurry, but the car must always adapt to this condition and keep the car stable. 


#### Data Filtering


Here, we will filter the images to keep the images where the car is in the middle of the road.   
I consider that tight to the left of the road = -1 , remains in the middle 0 and tight to the right = 1. 
So I filter to only have the data of the car when it is in the middle of the road.  


## Results
After 4 hours of training
<video controls src="Après 4h d’entrainement-1.mp4" title="Title"></video>

After 36 hours of Training 
<video controls src="Après 36h d’entrainement-1.mp4" title="Title"></video> 

Still a lot to do on tris project. 
