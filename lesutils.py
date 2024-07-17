from imgaug.augmenters.meta import Sequential
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import os 
from sklearn.utils import shuffle 
import matplotlib.image as mpimg # it gives an RBG image for our training we don't use cv2 bse it gives GBR
from imgaug import augmenters as iaa 
import cv2 
import random
from tensorflow.keras.models import Sequential   
from tensorflow.keras.layers import Convolution2D, Flatten, Dense 
from tensorflow.keras.optimizers import Adam 

def getName(filepath): # split lec chemin d'access et garder que le nom de l'image 
    return filepath.split('\\')[-1] 

def importDataInfo(path): # tjrs en importation de données
    coloums = ['Center', 'Left', 'Right', 'Steering', 'Throttle', 'Brake', 'Speed'] # les colonnes dans mon excel (driving_log.csv)
    data = pd.read_csv(os.path.join(path, 'driving_log.csv'), names = coloums) #je prends bien les colones  
    #print(data.head()) 
    #print(data['Center'][0]) 
    #print(getName(data['Center'][0])) 
    data['Center'] =  data['Center'].apply(getName) # split lec chemin d'access et garder que le nom des tous les images du centre 
    #print(data.head()) 
    print('Total Images Imported:', data.shape[0]) # nmbr d'images importé du centre (on utilisera que les images du centre)
    return data

def balanceData(data, display = True): # we try to move straightin the middle of the road
    nBins = 31 #odd number so we could have the 0 at the center the have a + side and a - side)
    samplesPerBin= 1000
    hist, bins = np.histogram(data['Steering'], nBins) # how much data do we have of each class 
    #                         values of steering angles
    #print(bins)

    if display: 
        center = (bins[:-1] + bins[1:]) * 0.5 # to see how many values of 0 we have bse we want&hope to drive straight
        #print(center) # we will then see a 0 
        plt.bar(center, hist, width = 0.06)
        plt.plot((-1,1),(samplesPerBin, samplesPerBin)) 
        #        range of x axis, range of y axis (from 1000 to 1000)
        plt.show() 

    removeIndexList = [] # remove redondent data (if it exceeds 1000 samplesperBin)
    for j in range(nBins): 
        binDataList = [] 
        for i in range(len(data['Steering'])): 
            if data['Steering'][i] >= bins[j] and data['Steering'][i] <= bins[j+1]: 
                binDataList.append(i) 
        binDataList = shuffle(binDataList) # mélanger les données avant d'enlever celle qu'on veut pas, pour eviter d'enlver qu'un seuls types de données
        binDataList = binDataList[samplesPerBin:] #and we will obtain 1000 samplesperbin
        removeIndexList.extend(binDataList) 
    print('Removed Images', len(removeIndexList)) 
    data.drop(data.index[removeIndexList], inplace = True) # enlever les unwanted
    print('Remaining Images:', len(data))  

    if display: 
        hist, _ = np.histogram(data['Steering'], nBins) 
        center = (bins[:-1] + bins[1:]) * 0.5 
        plt.bar(center, hist, width = 0.06)
        plt.plot((-1,1),(samplesPerBin, samplesPerBin)) 
        plt.show() 

    return data 

def loadData(path, data): 
    imagesPath = [] 
    steering = [] 

    for i in range (len(data)): 
        indexedData = data.iloc[i]
        #print(indexedData) 
        imagesPath.append(os.path.join(path,'IMG', indexedData[0])) 
        #print(os.path.join(path,'IMG', indexedData[0])) 
        steering.append(float(indexedData[3])) 

    imagesPath = np.asarray(imagesPath)
    steering = np.asarray(steering) 
    return imagesPath, steering 
    
def augmentImage(imgPath, steering): # pour augmenter les images, on le flip et tourner dans tous les sens pour essayer d'avoir les trait (route) droit (se décaler vers le milieu de la route)
    img = mpimg.imread(imgPath)
    print(np.random.rand(), np.random.rand(), np.random.rand(), np.random.rand())
    #PAN #Le SDK PAN-OS pour Python (pan-OS-PYTHON) est un progiciel permettant 
    # d’interagir avec les périphériques Palo Alto Networks (y compris les pare-feu de nouvelle génération physiques et virtualisés et Panorama).
    if np.random.rand() < 0.5: # 50% chances of a image to be balanced 
        pan = iaa.Affine(translate_percent = {'x':(-0.1, 0.1), 'y':(-0.1, 0.1)})
        img = pan.augment_image(img) 

    #ZOOM 
    if np.random.rand() < 0.5: # 50% chances of a image to be balanced
        zoom = iaa.Affine(scale = (1, 1.2))
        img =  zoom.augment_image(img)

    #Brightness 
    if np.random.rand() < 0.5: # 50% chances of a image to be balanced
        brightness = iaa.Multiply((0.2, 1.2)) 
        img = brightness.augment_image(img) 

    #FLIP
    if np.random.rand() < 0.5: # 50% chances of a image to be balanced
        img = cv2.flip(img, 1)  
        steering = -steering 
     
    return img, steering 

# imgRe, st = augmentImage('test.jpg', 0) 
# plt.imshow(imgRe)
# plt.show() 

def preProcessing(img): 
    img = img[60:135, :, :] # cropping the image so we will not see a part of the car or trees in the image
    img = cv2. cvtColor(img, cv2.COLOR_RGB2YUV) # change the color to better visualise the road 
    img = cv2.GaussianBlur(img,(3,3),0) # 3x3 kernel size 
    img = cv2.resize(img, (200, 66))
    img = img /255 

    return img 

"""#WILLCROP THE IMAGE ON A GRAPH 
# imgRe = preProcessing(mpimg.imread('test.jpg')) # cropping the image 
# plt.imshow(imgRe)
# plt.show() 
"""
imgRe = preProcessing(mpimg.imread('/mydrive/Project_2_autonomous-car/myData/test.jpg')) # cropping the image #############
plt.imshow(imgRe) #############
plt.show() #############

def batchGen(imagesPath, steeringList, batchSize, trainFlag): #generation de batches
    while True:
        imgBatch = []
        steeringBatch = []

        for i in range(batchSize): 
            index =  random.randint(0, len(imagesPath)- 1) 
            if trainFlag: 
                img, steering = augmentImage(imagesPath[index], steeringList[index])
            else: 
                img = mpimg.imread(imagesPath[index]) 
                steering = steeringList[index]

            img = preProcessing(img) 
            imgBatch.append(img)
            steeringBatch.append(steering)

        yield(np.asarray(imgBatch), np.asarray(steeringBatch)) 

def createModel(): 
    model = Sequential() 

    model.add(Convolution2D(24,(5,5),(2,2), input_shape = (66, 200, 3), activation = 'elu')) 
    #               filter, kernel size, stride
    model.add(Convolution2D(36,(5,5),(2,2), activation = 'elu')) 
    model.add(Convolution2D(48,(5,5),(2,2), activation = 'elu')) 
    model.add(Convolution2D(64,(3,3), activation = 'elu')) 
    model.add(Convolution2D(64,(3,3), activation = 'elu')) 

    model.add(Flatten()) 
    model.add(Dense(100,activation = 'elu'))
    model.add(Dense(50,activation = 'elu'))
    model.add(Dense(10,activation = 'elu'))
    model.add(Dense(1)) 

    model.compile(Adam(lr = 0.0001), loss = 'mse') 

    return model 