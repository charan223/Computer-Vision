import sys
sys.path.append('/usr/lib/python2.7/dist-packages')
import numpy as np
import cv2
import random
from matplotlib import pyplot as plt
from math import log
import math

# Load image 1
cap = cv2.imread('cap.bmp',0)
cv2.imshow('cap',cap)
cv2.waitKey(0)
# Load image 2
lego = cv2.imread('lego.tif',)
cv2.imshow('lego',lego)
cv2.waitKey(0)

# Convert the image to greyscale
graylego = cv2.imread('lego.tif',0)

'''
cv2.imshow('image',img)
cv2.waitKey(0)

# convert pixel to grayscale
def weightedAverage(pixel):
    return 0.299*pixel[0] + 0.587*pixel[1] + 0.114*pixel[2]

# convert color image to grayscale
for rownum in range(len(img)):
    for colnum in range(len(img[rownum])):
         img[rownum,colnum] = weightedAverage(img[rownum,colnum])

# show grayscale converted image
cv2.imshow('image',img)
cv2.waitKey(0)
'''


def gaussian_noise(image,s):
	gauss = np.random.normal(0,s,image.shape)
	gauss = gauss.reshape(image.shape)
	info = np.iinfo(image.dtype) 
	image1= image.astype(np.float) / info.max 	
	noisy = image1 + gauss
	return noisy


def median_filter(noisy):
    	final1 = noisy[:]
    	for y in range(1,noisy.shape[0]-1):
       	 	for x in range(1,noisy.shape[1]-1):
            		final1[y,x]=noisy[y,x]

   	members=[noisy[0,0]]*9
    	for y in range(1,noisy.shape[0]-1):
        	for x in range(1,noisy.shape[1]-1):
            		members[0] = noisy[y-1,x-1]
            		members[1] = noisy[y,x-1]
            		members[2] = noisy[y+1,x-1]
            		members[3] = noisy[y-1,x]
            		members[4] = noisy[y,x]
            		members[5] = noisy[y+1,x]
            		members[6] = noisy[y-1,x+1]
            		members[7] = noisy[y,x+1]
            		members[8] = noisy[y+1,x+1]
            		members.sort()
            		final1[y,x]=members[4]
	return final1

def mean_filter(noisy):
    	final = noisy[:]
   	for y in range(1,noisy.shape[0]-1):
       	 	for x in range(1,noisy.shape[1]-1):
         		final[y,x]=noisy[y,x]

    	members=[noisy[0,0]]*9
   	for y in range(1,noisy.shape[0]-1):
        	for x in range(1,noisy.shape[1]-1):
			members[0] = noisy[y-1,x-1]
			members[1] = noisy[y,x-1]
			members[2] = noisy[y+1,x-1]
			members[3] = noisy[y-1,x]
			members[4] = noisy[y,x]
			members[5] = noisy[y+1,x]
			members[6] = noisy[y-1,x+1]
			members[7] = noisy[y,x+1]
			members[8] = noisy[y+1,x+1]
			final[y,x]=(members[0]+members[1]+members[2]+members[3]+members[4]+members[5]+members[6]+members[7]+members[8])/9
	return final

def filtering(OriginalImage,s):

	print "s =",s
	noisyimg = gaussian_noise(OriginalImage,s)

	medianfilteredimg = median_filter(noisyimg)
	# cv2.imshow('Median Filtered Image',medianfilteredimg)
	# cv2.waitKey(0)

	meanfilteredimg = mean_filter(noisyimg)
	# cv2.imshow('Mean Filtered Image',meanfilteredimg)
	# cv2.waitKey(0)

	#mse = ((OriginalImage - medianfilteredimg) ** 2).mean(axis=None)
	mse = np.sum((OriginalImage.astype("float") - medianfilteredimg.astype("float")) ** 2)
	mse /= float(OriginalImage.shape[0] * OriginalImage.shape[1])
	psnr = 20* log(255,10) - 10*log(mse,10)
	print "Median PSNR =",psnr

	#mse1 = ((OriginalImage - meanfilteredimg) ** 2).mean(axis=None)
	mse1 = np.sum((OriginalImage.astype("float") - meanfilteredimg.astype("float")) ** 2)
	mse1 /= float(OriginalImage.shape[0] * OriginalImage.shape[1])
	psnr1 = 20* log(255,10) - 10*log(mse1,10)
	print "PMean SNR =",psnr1
	
	librarymeanfilteredimg = cv2.blur(noisyimg,(5,5))
	librarymedianfilteredimg =cv2.medianBlur(noisyimg,1)

	mse = np.sum((OriginalImage.astype("float") - librarymedianfilteredimg.astype("float")) ** 2)
	mse /= float(OriginalImage.shape[0] * OriginalImage.shape[1])
	psnr = 20* log(255,10) - 10*log(mse,10)
	print "Median PSNR =",psnr

	#mse1 = ((OriginalImage - meanfilteredimg) ** 2).mean(axis=None)
	mse1 = np.sum((OriginalImage.astype("float") - librarymeanfilteredimg.astype("float")) ** 2)
	mse1 /= float(OriginalImage.shape[0] * OriginalImage.shape[1])
	psnr1 = 20* log(255,10) - 10*log(mse1,10)
	print "PMean SNR =",psnr1

	titles = ['Original Image', 'Added Noise Image',
	            'Median filtering', 'Mean Filtering']
	images = [OriginalImage, noisyimg, medianfilteredimg, meanfilteredimg]

	for i in xrange(4):
	    plt.subplot(2,2,i+1),plt.imshow(images[i],'gray')
	    plt.title(titles[i])
	    plt.xticks([]),plt.yticks([])

	plt.savefig("Noise Filtering.png")
	plt.show()

	titles1 = ['Original Image', 'Added Noise Image',
	            'Library Median filtering', 'Library Mean Filtering']
	images1 = [OriginalImage, noisyimg, librarymedianfilteredimg, librarymeanfilteredimg]

	for i in xrange(4):
	    plt.subplot(2,2,i+1),plt.imshow(images1[i],'gray')
	    plt.title(titles1[i])
	    plt.xticks([]),plt.yticks([])

	plt.savefig("Library Noise Filtering.png")


if __name__ == "__main__":
	s=input("Give the value of standard deviation\n")
	filtering(cap,s)
