# Hand-Bone-Age-Analysis
Instructions to run this Repo are given below.

### Goal of the project

Hand bone age analysis project was conducted to deploy a machine learning model capable of predicting the age by only analysing the X-ray of hand bone. Prediction of bone age is very important in the field of bone health. The bone age can help evaluate how fast or slowly a child's skeleton is maturing, which can help doctors diagnose conditions that delay or accelerate physical growth and development.

Bone age can be used to predict:

* How much time a child will be growing
* When a child will enter puberty
* What the child's ultimate height will be

It can further be used to monitor progress and guide treatment of kids with conditions that affect growth, including:

* Diseases that affect the levels of hormones involved in growth, such as growth hormone deficiency, hypothyroidism, precocious puberty, and adrenal gland disorders
* Genetic growth disorders, such as Turner syndrome (TS)
* Orthopedic or orthodontic problems in which the timing and type of treatment (surgery, bracing, etc.) must be guided by the child's predicted growth

Various data mining and machine learning techniques were employed to obtain an accurate prediction model.


### Implementation

The RSNA Bone Age dataset was used to train our model. The dataset contained around 12k images of X-rays of hand bone of children.
Spyder notebook was used for writing all code and analysis of the images and other mathematical charts.

#### 1) Pre-processing :

Initially all the csv files and imagess were loaded. Various pre-processing techniques were applied to the loaded images for better results overall. 

OpenCV library was used for dealing with images.

1) Images were enhanced using Histogram Equalization. This technique makes the important detaills of an image brighter for better training.

2) Morphological Operations such as dilation and erosion were applied for extracting only bone details from the entire hand X-ray.
Following figures indicated how this was proceeded with:


<img src="/Image_samples/0.Original_Image.png" width="350"> -----------------><img src="/Image_samples/1.segment_hand&fingers.png" width="350">
                                                               
      
<img src="/Image_samples/2.hand_silhoutte.png" width="350"> -----------------><img src="/Image_samples/3.fingers_only.png" width="350"> <img src="/Image_samples/4.fingers_only_equalized.png" width="350">

The First image is the original X-ray of child. The second images is segmentation of hand and fingers. The third image is hand silhoutte. Thr fourth image is fingers(bone) only. The last image is enhanced version of previous image.


#### 2) Model:

Two different neural networks were trained initially and later both of them were combined into one neural network to obtain the final result. The first is a Convolutional Neural Network (CNN) which is used to train the images, and the second one is a Multi Layer Perceptron (MLP) used for training of CSV data(Gender in our case). The CNN has filters as (16,32,64) and 1 hidden layer. The MLP has 2 hidden layers. Relu activation was used in the neural networks. 30 Epochs were performed for the training of the combined neural network.

#### 3)Prediction and evaluation of the model:

After the model was trained, prediction was done for test data to see how the data performs on unseen data images. Various statistical measures were calculated for evaluation. This included mean, standard deviation, mean absolute error etc.



