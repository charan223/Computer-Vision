from __future__ import print_function
import sys
sys.path.append('/usr/lib/python2.7/dist-packages')
import numpy as np
import cv2
import math
import sys

if (len(sys.argv) < 2):
    print("Pass the path to input file as an argument")
    exit()
input_image_file = sys.argv[1]

cap = cv2.imread(input_image_file,0)

threshold1=-20
threshold2=0
threshold3=0
#f= open("check.txt","w")
def coeff(x):
	if x==0:
		return 1.0/math.sqrt(2)
	return 1.0

def block_dct (Pixel):
	# block size
	N = 8 
	output = np.zeros(Pixel.shape,np.int32)
	for u in range(N):
	    for v in range(N):
	        temp = 0.0
	        for x in range(N):
			    for y in range(N):
			    	temp += Pixel[x,y] * math.cos(v*math.pi*(y+0.5)/N) * math.cos(u*math.pi*(x+0.5)/N)
	        temp *= coeff(v) * coeff(u)
	        temp = temp / 4
	        # DCT[i][j] = INT_ROUND(temp);
		if(u+v<6 and temp<threshold1):
			output[u,v]=0
		elif(u+v<10 and u+v>5 and temp<threshold2):
			output[u,v]=0
		elif(u+v>10 and temp<threshold3):
			output[u,v]=0
	        else:
			output[u,v] = temp
	return output

def inverse_block_dct(Pixel):
	# block size
	N = 8 
	
	output = np.zeros(Pixel.shape,np.uint32)
	for i in range(N):
	    for j in range(N):
	        temp = 0.0
	        for u in range(N):
			    for v in range(N):
			    	temp += coeff(u) * coeff(v) * Pixel[u,v] * math.cos(u*math.pi*(i+0.5)/N) * math.cos(v*math.pi*(j+0.5)/N)
	        temp = temp / 4
	        # DCT[i][j] = INT_ROUND(temp);
	        output[i,j] = temp
	return output

# edge()

# print "Original Image"
# print cap[0:8,0:8]
# test = block_dct(cap[0:8,0:8])
# print test
# inversetest = inverse_block_dct(test)
# print inversetest


def dct(image):
	output = np.zeros(image.shape,np.int)

	# divide image into 8*8 blocks (partitioning)
	windowsize_r = 8
	windowsize_c = 8

	for r in range(0,image.shape[0] - windowsize_r+1, windowsize_r):
		for c in range(0,image.shape[1] - windowsize_c+1, windowsize_c):
			window = image[r:r+windowsize_r,c:c+windowsize_c]
			test = block_dct(window)
			#f.write("\n\n\n\n\n")
			for i in range(windowsize_r):
				for j in range(windowsize_c):
					output[r+i,c+j] = test[i,j]
	return output


def inv_dct(image):
	output = np.zeros(image.shape,np.uint8)

	# divide image into 8*8 blocks
	windowsize_r = 8
	windowsize_c = 8

	for r in range(0,image.shape[0] - windowsize_r+1, windowsize_r):
		for c in range(0,image.shape[1] - windowsize_c+1, windowsize_c):
			window = image[r:r+windowsize_r,c:c+windowsize_c]
			test = inverse_block_dct(window)
			for i in range(windowsize_r):
				for j in range(windowsize_c):
					output[r+i,c+j] = test[i,j]
	return output

def dct_transform():

	# DCT TRansform
	
	block_dct_transformed_image = dct(cap)
	#print block_dct_transformed_image
	cv2.imwrite("DCT Transformed.png",block_dct_transformed_image)
	cv2.imshow('Block transformed image',block_dct_transformed_image)
	# Inverse DCT Transform
	
	after_inverse_dct = inv_dct(block_dct_transformed_image)
	#print after_inverse_dct
	cv2.imwrite("After inverse transform.png",after_inverse_dct)
	cv2.imshow('After inverse transform',after_inverse_dct)
	cv2.waitKey(0)

if __name__ == "__main__":
	dct_transform()
	#dct(cap)
