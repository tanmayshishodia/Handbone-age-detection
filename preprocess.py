import cv2
import numpy as np

def pre_process(img):
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
    #cv2.imwrite('pre/fingers_only.png',fingers);
    #cv2.waitKey(0)
    return (fingers)
    



