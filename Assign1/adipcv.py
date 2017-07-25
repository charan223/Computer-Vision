import numpy as np
import cv2




def display(img1,img2):
	cv2.imshow('image',img1)
	cv2.waitKey(1000) 
	cv2.destroyWindow('image')
	cv2.imshow('image',img2)
	cv2.waitKey(1000) 
	cv2.destroyWindow('image')
	return


def displaygray(img3,img4):
	cv2.imshow('image',img3)
	cv2.waitKey(1000) 
	cv2.destroyWindow('image')
	cv2.imshow('image',img4)
	cv2.waitKey(1000) 
	cv2.destroyWindow('image')
	return


def addnoiseandfilter(image,s):
	gauss = np.random.normal(0,s,image.shape)
	gauss = gauss.reshape(image.shape)
	info = np.iinfo(image.dtype) 
	image1= image.astype(np.float) / info.max 	
	noisy = image1 + gauss

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

	cv2.imshow('mean filtered image',final)
	cv2.waitKey(2000) 
	cv2.destroyWindow('mean filtered image')

	blur = cv2.blur(noisy,(5,5))
	cv2.imshow('function mean filtered image',blur)
	cv2.waitKey(2000) 
	cv2.destroyWindow('function mean filtered image')


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


	cv2.imshow('median filtered image',final1)
	cv2.waitKey(2000) 
	cv2.destroyWindow('median filtered image')

	yt=cv2.medianBlur(noisy,1)
	cv2.imshow('function median filtered image',yt)
	cv2.waitKey(2000) 
	cv2.destroyWindow('function median filtered image')
	return 	





#Load an color image
img1 = cv2.imread('cap.bmp')
img2 = cv2.imread('lego.tif')


img3 = cv2.cvtColor( img1, cv2.COLOR_RGB2GRAY )
cv2.imwrite( "cap-gray.bmp", img1 )

img4 = cv2.cvtColor( img2, cv2.COLOR_RGB2GRAY )
cv2.imwrite( "lego-gray.tif", img2 )
#display(img1,img2)
#displaygray(img3,img4)


s = input('Enter Standard Deviation of Gaussian Blurr')
addnoiseandfilter(img3,s)

