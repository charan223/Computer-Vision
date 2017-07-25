from cv2 import * #Import functions from OpenCV
import cv2
import numpy as np


if __name__ == '__main__':
    source = cv2.imread("cap.bmp", CV_LOAD_IMAGE_GRAYSCALE)
    gauss = np.random.normal(0,0.5,source.shape)
    gauss = gauss.reshape(source.shape)
    info = np.iinfo(source.dtype) 
    image1= source.astype(np.float) / info.max 	
    noisy = image1 + gauss
    cv2.imshow('check',noisy)
    cv2.waitKey(2000) 
    cv2.destroyWindow('check')
    final = noisy[:]
    z=len(noisy)-1
    for y in range(0,z):
        for x in range(0,y):
            final[y,x]=noisy[y,x]

    members=[noisy[0,0]]*9
    members1=[noisy[0,0]]*6
    members2=[noisy[0,0]]*4


    for y in range(0,z):
        for x in range(0,y):
        	if y-1<0 and x-1>=0 and x+1<=z and y+1<=z :
        		members1[0] = noisy[y,x-1]
            		members1[1] = noisy[y,x+1]
		    	members1[2] = noisy[y+1,x]
		    	members1[3] = noisy[y+1,x+1]
		    	members1[4] = noisy[y+1,x-1]
		    	members1[5] = noisy[y,x]
		    	members1.sort()
		    	final[y,x]=(members1[2]+members1[3])/2
		elif y-1>=0 and x-1>=0 and x+1<=z and y+1>z:
        		members1[0] = noisy[y,x-1]
		    	members1[1] = noisy[y,x+1]
		    	members1[2] = noisy[y-1,x]
		    	members1[3] = noisy[y-1,x+1]
		    	members1[4] = noisy[y-1,x-1]
		    	members1[5] = noisy[y,x]
		    	members1.sort()
		    	final[y,x]=(members1[2]+members1[3])/2
        	elif y-1>=0 and x-1<0 and x+1<=z and y+1<=z:
        		members1[0] = noisy[y+1,x]
		    	members1[1] = noisy[y-1,x]
		    	members1[2] = noisy[y,x+1]
		    	members1[3] = noisy[y+1,x+1]
		    	members1[4] = noisy[y-1,x+1]
		    	members1[5] = noisy[y,x]
		    	members1.sort()
		    	final[y,x]=(members1[2]+members1[3])/2
        	elif y-1>=0 and x-1>=0 and x+1>z and y+1<=z:
        		members1[0] = noisy[y+1,x]
		    	members1[1] = noisy[y-1,x]
		    	members1[2] = noisy[y,x-1]
		    	members1[3] = noisy[y+1,x-1]
		    	members1[4] = noisy[y-1,x-1]
		    	members1[5] = noisy[y,x]
		    	members1.sort()
		    	final[y,x]=(members1[2]+members1[3])/2
        	elif y-1<0 and x-1<0 and x+1<=z and y+1<=z:
        		members2[0] = noisy[y,x]
		    	members2[1] = noisy[y+1,x]
		    	members2[2] = noisy[y,x+1]
		    	members2[3] = noisy[y+1,x+1]
		    	members2.sort()
		    	final[y,x]=(members1[1]+members1[2])/2
        	elif y-1>=0 and x-1>=0 and x+1>z and y+1>z:
        		members2[0] = noisy[y,x]
		    	members2[1] = noisy[y-1,x]
		    	members2[2] = noisy[y,x-1]
		    	members2[3] = noisy[y-1,x-1]
		    	members2.sort()
		    	final[y,x]=(members1[1]+members1[2])/2
        	elif y-1>=0 and x-1<0 and x+1<=z and y+1>z:
        		members2[0] = noisy[y,x]
		    	members2[1] = noisy[y-1,x]
		    	members2[2] = noisy[y,x+1]
		    	members2[3] = noisy[y-1,x+1]
		    	members2.sort()
		    	final[y,x]=(members1[1]+members1[2])/2            	
        	elif y-1<0 and x-1>=0 and x+1>z and y+1<=z :
        		members2[0] = noisy[y,x]
		    	members2[1] = noisy[y+1,x]
		    	members2[2] = noisy[y,x-1]
		    	members2[3] = noisy[y+1,x-1]
		    	members2.sort()
		    	final[y,x]=(members1[1]+members1[2])/2
		else:
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
		    	final[y,x]=members[4]


    cv2.imshow('check',noisy)
    cv2.waitKey(2000) 
    cv2.destroyWindow('check')
    cv2.imshow('check',final)
    cv2.waitKey(2000) 
    cv2.destroyWindow('check')

