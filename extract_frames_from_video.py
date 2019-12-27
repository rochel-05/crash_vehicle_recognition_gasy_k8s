import cv2
import matplotlib.pyplot as plt
import numpy as np
import math
import pandas as pd
from skimage.transform import resize
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from os import listdir
import os, glob

data_dir = "data/"

generated_test_path = "generated_frames_test/"
generated_train_raw_path = "generated_frames_train_raw/"

n_classes = 5
listVideo = []

def extract_frames_from_video_valid(srcDir):
    if glob.glob("static/generated_frames_valid/valid0.jpg"):
        for elt in os.listdir("static/generated_frames_valid/"):
            os.remove("static/generated_frames_valid/" + elt)
        print("Deletion files with sucess!!!")
        for video in os.listdir(srcDir):
            dir = srcDir
        file3 = dir + video
        count = 0
        cap = cv2.VideoCapture(file3)
        frameRate = cap.get(5)
        while(cap.isOpened()):
            frameId = cap.get(1)
            ret, frame = cap.read()
            if(ret!=True):
                break
            if(frameId%math.floor(frameRate)==0):
                filename = "valid%d.jpg"%count
                count+=1
                print(filename)
                cv2.imwrite("static/generated_frames_valid/" + filename, frame)
        cap.release()
        for video in os.listdir(srcDir):
            os.remove(srcDir+video)
    else:
        count = 0
        for video in os.listdir(srcDir):
            dir = srcDir
        file3 = dir + video
        cap = cv2.VideoCapture(file3)
        frameRate = cap.get(5)
        while(cap.isOpened()):
            frameId = cap.get(1)
            ret, frame = cap.read()
            if(ret!=True):
                break
            if(frameId%math.floor(frameRate)==0):
                filename = "valid%d.jpg"%count
                count+=1
                print(filename)
                cv2.imwrite("static/generated_frames_valid/" + filename, frame)
        cap.release()
        for video in os.listdir(srcDir):
            os.remove(srcDir + video)
    print("Done!!!")

def extract_frames_from_video_valid_scraping(url):
    if glob.glob("static/generated_frames_valid/valid0.jpg"):
        for elt in os.listdir("static/generated_frames_valid/"):
            os.remove("static/generated_frames_valid/" + elt)
        print("Deletion files with sucess!!!")
        file3 = url
        count = 0
        cap = cv2.VideoCapture(file3)
        frameRate = cap.get(5)
        while(cap.isOpened()):
            frameId = cap.get(1)
            ret, frame = cap.read()
            if(ret!=True):
                break
            if(frameId%math.floor(frameRate)==0):
                filename = "valid%d.jpg"%count
                count+=1
                print(filename)
                cv2.imwrite("static/generated_frames_valid/" + filename, frame)
        cap.release()
    else:
        count = 0
        file3 = url
        cap = cv2.VideoCapture(file3)
        frameRate = cap.get(5)
        while(cap.isOpened()):
            frameId = cap.get(1)
            ret, frame = cap.read()
            if(ret!=True):
                break
            if(frameId%math.floor(frameRate)==0):
                filename = "valid%d.jpg"%count
                count+=1
                print(filename)
                cv2.imwrite("static/generated_frames_valid/" + filename, frame)
        cap.release()
    print("Done!!!")

def visualize_frame_from_video_valid():
    img = plt.imread("generated_frames_test/valid0.jpg")
    plt.imshow(img)
    plt.show()

def load_data_train_raw():
    Xtrain = []
    Ximage=[]
    data = pd.read_csv(data_dir + "train_raw.csv")
    print(data.head())
    #load image
    for img_name in data.Image_ID:
        image = plt.imread(generated_train_raw_path + img_name)
        Xtrain.append(image)
    Xtrain = np.array(Xtrain)
    for i in range(0, Xtrain.shape[0]):
        image = resize(Xtrain[i], preserve_range=True, output_shape=(224, 224)).astype(int)
        Ximage.append(image)
    XTrain = np.array(Ximage)
    #load labels
    Ytrain = data.Class
    Ytrain = to_categorical(Ytrain, num_classes=n_classes)
    trainX, testX, trainY, testY = train_test_split(XTrain, Ytrain, random_state=0, test_size=0.2)

    trainX = np.reshape(trainX, (-1, 150528))
    testX = np.reshape(testX, (-1, 150528))
    return trainX, testX, trainY, testY

def load_data_test():
    Xtest = []
    Ximage=[]
    data = pd.read_csv(data_dir + "test.csv")
    print(data.head())
    #load image
    for img_name in data.Image_ID:
        image = plt.imread(generated_test_path + img_name)
        Xtest.append(image)
    Xtest = np.array(Xtest)
    for i in range(0, Xtest.shape[0]):
        image = resize(Xtest[i], preserve_range=True, output_shape=(224, 224)).astype(int)
        Ximage.append(image)
    XTest = np.array(Ximage)
    #load labels
    Ytest = data.Class
    Ytest = to_categorical(Ytest, num_classes=n_classes)
    trainX, testX, trainY, testY = train_test_split(XTest, Ytest, random_state=0, test_size=0.2)

    trainX = np.reshape(trainX, (-1, 150528))
    testX = np.reshape(testX, (-1, 150528))
    return trainX, testX, trainY, testY

def load_data_valid():
    X_valid = []
    X_valid_im = []
    #image train
    for img_name in listdir("static/generated_frames_valid/"):
        img = plt.imread("static/generated_frames_valid/" + img_name)
        X_valid.append(img)
    X_valid = np.array(X_valid)
    for i in range(0, X_valid.shape[0]):
        img = resize(X_valid[i], preserve_range=True, output_shape=(224, 224)).astype(int)
        X_valid_im.append(img)
    X_valid_im = np.array(X_valid_im)
    return X_valid_im