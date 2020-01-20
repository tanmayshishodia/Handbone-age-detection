'''import plaidml.keras
plaidml.keras.install_backend()
'''


import pandas as pd
import numpy as np

from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras.layers import Flatten
from keras.layers import Input
from keras.models import Model
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import cv2
from matplotlib import pyplot as plt



'''
im=cv2.imread('boneage-training-dataset/boneage-training-dataset/4007.png',0)
im=cv2.resize(im,(256,256))
print(im.shape)

equ=cv2.equalizeHist(im)

cv2.imshow('before',im)
cv2.waitKey(0)

crop=im[0:224,10:244]

cv2.imshow('after',crop)
cv2.waitKey(0)
'''

df_train=pd.read_csv('boneage-training-dataset.csv')
df_test=pd.read_csv('boneage-test-dataset.csv')




def pre_process(img,ind):
    im=img
    #cv2.imshow('before',im)
    #cv2.waitKey(0)
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    #cv2.imshow('before',gray)
    #cv2.waitKey(0)
    gray_silhoutte=gray


    #func1
    kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3),(1,1))
    gray = cv2.morphologyEx(gray, cv2.MORPH_ELLIPSE, kernel)
    
    adaptiveMethod =cv2.ADAPTIVE_THRESH_GAUSSIAN_C
    gray=cv2.adaptiveThreshold(gray,255,adaptiveMethod,cv2.THRESH_BINARY,9,-5)
    
    dilate_sz=1
    element=cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*dilate_sz, 2*dilate_sz),(dilate_sz,dilate_sz))
    gray=cv2.dilate(gray,element)
    
    
    #cv2.imshow('Hand+Fingers',gray)
    #cv2.waitKey(0)
    #cv2.imwrite('hand_fingers.png',gray)
    
    
    #func2
    kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7),(3,3))
    gray_silhoutte = cv2.morphologyEx(gray_silhoutte, cv2.MORPH_ELLIPSE, kernel)
    
    adaptiveMethod =cv2.ADAPTIVE_THRESH_MEAN_C
    gray_silhoutte=cv2.adaptiveThreshold(gray_silhoutte,255,adaptiveMethod,cv2.THRESH_BINARY,251,5)
    
    erode_sz=5
    element=cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*erode_sz+1, 2*erode_sz+1),(erode_sz,erode_sz))
    gray_silhoutte=cv2.erode(gray_silhoutte,element)
    
    dilate_sz=1
    element=cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*dilate_sz+1, 2*dilate_sz+1),(dilate_sz,dilate_sz))
    gray_silhoutte=cv2.dilate(gray_silhoutte,element)
    
    gray_silhoutte=cv2.bitwise_not(gray_silhoutte)
    
    
    #cv2.imshow('Hand',gray_silhoutte)
    #cv2.waitKey(0)
    #cv2.imwrite('hand_silhoutte.png',gray_silhoutte)
    
    fingers=gray-gray_silhoutte
    #cv2.imshow('fingers',fingers)
    cv2.imwrite('pre/'+str(ind)+'.png',cv2.cvtColor(fingers, cv2.COLOR_GRAY2BGR))
    #cv2.waitKey(0)
    return (cv2.cvtColor(fingers, cv2.COLOR_GRAY2BGR))
    


#For loading all images
def get_images(in_path):
    images=[]
    for i in range(7000):
        ind=df_train['id'][i]
        path=in_path+'/'+str(ind)+'.png'
        image=cv2.imread(path)
        R, G, B = cv2.split(image)
        output1_R = cv2.equalizeHist(R)
        output1_G = cv2.equalizeHist(G)
        output1_B = cv2.equalizeHist(B)
        equ = cv2.merge((output1_R, output1_G, output1_B))
        equ=pre_process(equ,ind)
        equ=cv2.resize(equ,(240,240))
        images.append(equ)
    return np.array(images)
    
    
    
# For creating CNN model    
def create_cnn(width, height, depth, filters=(16, 32, 64), regress=False):
	# initialize the input shape and channel dimension, assuming
	# TensorFlow/channels-last ordering
	inputShape = (height, width, depth)
	chanDim = -1

	# define the model input
	inputs = Input(shape=inputShape)

	# loop over the number of filters
	for (i, f) in enumerate(filters):
		# if this is the first CONV layer then set the input
		# appropriately
		if i == 0:
			x = inputs

		# CONV => RELU => BN => POOL
		x = Conv2D(f, (3, 3), padding="same")(x)
		x = Activation("relu")(x)
		x = BatchNormalization(axis=chanDim)(x)
		x = MaxPooling2D(pool_size=(2, 2))(x)

	# flatten the volume, then FC => RELU => BN => DROPOUT
	x = Flatten()(x)
	x = Dense(16)(x)
	x = Activation("relu")(x)
	x = BatchNormalization(axis=chanDim)(x)
	x = Dropout(0.5)(x)

	# apply another FC layer, this one to match the number of nodes
	# coming out of the MLP
	x = Dense(4)(x)
	x = Activation("relu")(x)

	# check to see if the regression node should be added
	if regress:
		x = Dense(1, activation="linear")(x)

	# construct the CNN
	model = Model(inputs, x)

	# return the CNN
	return model    
    

    
images=get_images('boneage-training-dataset/boneage-training-dataset')

 
#cv2.imshow('images',images[0])   # To display images which is loaded

# To scale down pixel intensities to [0,1]
images_1 = images/ 255.0
#images_2 = images[3000:6000] / 255.0
#images_3 = images[6000:8000] / 255.0
#images_4 = images[9000:12611] / 255.0




#Splitting into train | test
split = train_test_split(df_train[:7000], images_1, test_size=0.20, random_state=42)
(trainAttrX, testAttrX, trainImagesX, testImagesX) = split

#To scale down bone-age(years-ouput) to [0,1] for better results
max_age = trainAttrX["boneage"].max()
trainY = trainAttrX["boneage"] / max_age
testY = testAttrX["boneage"] / max_age


#Create CNN 
model = create_cnn(240, 240,3, regress=True)
opt = Adam(lr=1e-3, decay=1e-3 / 200)
model.compile(loss="mean_absolute_percentage_error", optimizer=opt)


#Training our model
print("Training model...")
model.fit(trainImagesX, trainY, validation_data=(testImagesX, testY),epochs=25, batch_size=8)


#Predicting
print("Predicting hand bone ages...")
preds = model.predict(testImagesX)
preds=preds*max_age
testY=testY*max_age
preds=preds/12
testY=testY/12
#Finding error statistics etc
diff = preds.flatten() - testY
percentDiff = (diff / testY) * 100
absPercentDiff = np.abs(percentDiff)
 
mean = np.mean(absPercentDiff)
std = np.std(absPercentDiff)





