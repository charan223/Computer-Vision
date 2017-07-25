from cv2 import * #Import functions from OpenCV
import cv2
import numpy as np

if __name__ == '__main__':
    source = cv2.imread("cap.bmp", CV_LOAD_IMAGE_GRAYSCALE)
    gauss = np.random.normal(0,0.6,source.shape)
    gauss = gauss.reshape(source.shape)
    info = np.iinfo(source.dtype) 
    image1= source.astype(np.float) / info.max 	
    noisy = image1 + gauss

    cv2.imshow('Source_Picture1', source) #Show the image
    cv2.imshow('Source_Picture', noisy) #Show the image

    cv2.waitKey()
