from PIL import Image
import sys
sys.path.append('/usr/local/lib/python2.7/dist-packages')
sys.path.append('/usr/lib/python2.7/dist-packages')
import cv2
import numpy as np
from matplotlib import pyplot as plt

# Configuration parameters are taken as command line arguments

# Parsing arguments

# Image 1 name
image_1_name = sys.argv[1]

# Image 2 name
image_2_name = sys.argv[2]

# Method used for finding corresponding points
# 1. sift
# 2. manual
method = sys.argv[3]

count1 = 1
count2 = 1

def drawMatches(img1, kp1, img2, kp2, matches):

    # Create a new output image that concatenates the two images together
    # (a.k.a) a montage
    rows1 = img1.shape[0]
    cols1 = img1.shape[1]
    rows2 = img2.shape[0]
    cols2 = img2.shape[1]

    out = np.zeros((max([rows1,rows2]),cols1+cols2,3), dtype='uint8')

    # Place the first image to the left
    out[:rows1,:cols1] = np.dstack([img1, img1, img1])

    # Place the next image to the right of it
    out[:rows2,cols1:] = np.dstack([img2, img2, img2])
	

    # For each pair of points we have between both images
    # draw circles, then connect a line between them
    img1_keypoints=[]
    img2_keypoints=[]
    for mat in matches:

        # Get the matching keypoints for each of the images
        img1_idx = mat.queryIdx
        img2_idx = mat.trainIdx

        # x - columns
        # y - rows
        (x1,y1) = kp1[img1_idx].pt
        (x2,y2) = kp2[img2_idx].pt
        img1_keypoints.append([y1,x1])
        img2_keypoints.append([y2,x2])

        # Draw a small circle at both co-ordinates
        # radius 4
        # colour blue
        # thickness = 1
        cv2.circle(out, (int(x1),int(y1)), 4, (0, 0, 255), 1)   
        cv2.circle(out, (int(x2)+cols1,int(y2)), 4, (0, 0, 255), 1)

        # Draw a line in between the two points
        # thickness = 1
        # colour blue
        cv2.line(out, (int(x1),int(y1)), (int(x2)+cols1,int(y2)), (0, 0, 255), 1)

    #print img1_keypoints,img2_keypoints
    # Show the image
    cv2.imshow('Matched Features', out)
    cv2.waitKey(0)
    cv2.destroyWindow('Matched Features')

    # Also return the image if you'd like a copy
    return out,img1_keypoints,img2_keypoints





def draw_circle(event,x,y,flags,param):
    global img1,img2, count1, count2, pts1, pts2
    if event == cv2.EVENT_LBUTTONDBLCLK:
        if param==1:
            cv2.putText(img1,str(count1),(x,y),cv2.FONT_HERSHEY_SIMPLEX, 1,(0,0,0),2)
            count1 +=1
            pts1.append((x,y))
        elif param==2:
            cv2.putText(img2,str(count2),(x,y),cv2.FONT_HERSHEY_SIMPLEX, 1,(0,0,0),2)
            count2 +=1
            pts2.append((x,y))

        print x,y

img1orig = cv2.imread(image_1_name,0)  #queryimage # left image
img2orig = cv2.imread(image_2_name,0) #trainimage # right image
#img1=Image.open(image_1_name)
#img2=Image.open(image_2_name)
# Resizing both the images

img1 = cv2.resize(img1orig, (700,700))
img2 = cv2.resize(img2orig, (700,700))

def save_points(pts1, pts2, filename1, filename2):
    np.save(filename1, pts1)
    np.save(filename2, pts2)

def load_points(filename1, filename2):
    pts1 = np.load(filename1)
    pts2 = np.load(filename2)
    return pts1, pts2

def getEpilines(pt, F):
    temp=np.ones((8,3))
    temp[:,:-1]=pt
    return np.dot(F, temp.T).T


def skew(a):
    return np.array([[0,-a[2],a[1]],[a[2],0,-a[0]],[-a[1],a[0],0]])

def getEpipoles(F):
    epipoles={}
    U,s,V=np.linalg.svd(F, full_matrices=False)
    epipoles['left']=U[:,-1]/U[:,-1][2]
    epipoles['right']=V[-1]/V[-1][2]
    return epipoles




def get_3D_coordinates(P,P_dash,img1_keypoints,img2_keypoints):
    print "\n\n"
    print "---------Calculating 3D points for 8 points we had chosen----"
    pts1=np.array(img1_keypoints)
    pts2=np.array(img2_keypoints)
    C_dash_=-np.dot(np.linalg.inv(P_dash[:,:3]),P_dash[:,-1])
    #C_dash_=C_dash_.reshape(3,1)
    C_dash=np.zeros(4,)
    C_dash[:3]=C_dash_
    C_dash[-1]=1.0
    C_dash=C_dash.reshape(4,1)
    C=np.array([0.0,0.0,0.0,1.0]).reshape(4,1)
    Depths=[]
    Co_ordinates=[]
    
    l=np.dot(np.linalg.pinv(P),np.array([pts1[0][0],pts1[0][1],1.0]).reshape(3,1))-np.dot(np.linalg.pinv(P_dash),np.array([pts2[0][0],pts2[0][1],1.0]).reshape(3,1))
    lambda1=(l[0][0]-l[3][0]*C_dash[0][0])/(C_dash[0][0]-C[0][0])
    for i in range(8):
        X=np.dot(np.linalg.pinv(P),np.array([pts1[i][0],pts1[i][1],1.0]).reshape(3,1))
        Y=np.array([0.0,0.0,0.0,lambda1]).reshape(4,1)
        Z=X+Y
        depth=1/(Z[-1]*np.sqrt(P_dash[2,0]**2+P_dash[2,1]**2+P_dash[2,2]**2))
        Z=Z/Z[-1]
        
        print "Co_ordinates of point " + str(i+1)+" is: ",Z[:3,0]
        print "Depth of point " + str(i+1)+" is: ",depth[0]
        print "\n"





good = []
pts1 = []
pts2 = []
ptsr = np.zeros((2,2)) 
ptsl = np.zeros((2,2)) 
#----------------------------------------#
# Findiing corresponding points
#----------------------------------------#

# Method 1: Using SIFT

#(i) PART
if (method=="sift"):

    '''sift = cv2.SIFT()
    # find the keypoints and descriptors with SIFT
	kp1, des1 = sift.detectAndCompute(img1,None)
	kp2, des2 = sift.detectAndCompute(img2,None)

    # FLANN parameters
	FLANN_INDEX_KDTREE = 0
	index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
	search_params = dict(checks=50)
	flann = cv2.FlannBasedMatcher(index_params,search_params)
	matches = flann.knnMatch(des1,des2,k=2)

    # ratio test as per Lowe's paper
	for i,(m,n) in enumerate(matches):
		if m.distance < 0.8*n.distance:
			good.append(m)
			pts2.append(kp2[m.trainIdx].pt)
			pts1.append(kp1[m.queryIdx].pt)
    # We select only inlier points
    # pts1 = pts1[mask.ravel()==1]
    # pts2 = pts2[mask.ravel()==1]
    '''
    orb = cv2.ORB(1000, 2.5)

	# Detect keypoints of original image
    (kp1,des1) = orb.detectAndCompute(img1, None)


	# Detect keypoints of rotated image
    (kp2,des2) = orb.detectAndCompute(img2, None)

	# Create matcher
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

	# Do matching
    matches = bf.match(des1,des2)

	# Sort the matches based on distance.  Least distance
	# is better
    matches = sorted(matches, key=lambda val: val.distance)

	# Show only the top 10 matches - also save a copy for use later
    out,pt1,pt2 = drawMatches(img1, kp1, img2, kp2, matches[:8])
    pts1=np.array(pt1)
    pts2=np.array(pt2)


# Method 2: Using manual selection

elif (method=="manual"):

    # Display both the images to manually select the corresponding points in them

    cv2.namedWindow('First Image')
    cv2.setMouseCallback('First Image',draw_circle,param=1)
    cv2.namedWindow('Second Image')
    cv2.setMouseCallback('Second Image',draw_circle,param=2)
    while(1):
        cv2.imshow('Second Image',img2)
        cv2.imshow('First Image',img1)
        if cv2.waitKey(20) & 0xFF == 27:
            break

    cv2.destroyAllWindows()

    save_points(pts1, pts2, "pts1","pts2")


pts1 = np.float32(pts1)
pts2 = np.float32(pts2)

#----------------------------------------#
# Computing fundamental Matrix
#----------------------------------------#

#(ii) PART
F, mask = cv2.findFundamentalMat(pts1,pts2,cv2.FM_LMEDS)
np.set_printoptions(suppress=True)
print "Fundamental Matrix is"
print F


"""print "Fundamental Matrix as computed by DLT is"
F2 = compute_fundamental(pts1, pts2)
print F2"""

# Visualizing epipolar lines of the feature points used in the computation

def drawlines(img1,img2,lines,pts1,pts2):
    ''' img1 - image on which we draw the epilines for the points in img2
        lines - corresponding epilines '''
    print img1.shape
    r,c = img1.shape
    #img1 = cv2.cvtColor(img1,cv2.COLOR_GRAY2BGR)
    #img2 = cv2.cvtColor(img2,cv2.COLOR_GRAY2BGR)
    for r,pt1,pt2 in zip(lines,pts1,pts2):
        color = tuple(np.random.randint(0,255,3).tolist())
        x0,y0 = map(int, [0, -r[2]/r[1] ])
        x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
        img1 = cv2.line(img1, (x0,y0), (x1,y1), color,1)
        img1 = cv2.circle(img1,tuple(pt1),5,color,-1)
        img2 = cv2.circle(img2,tuple(pt2),5,color,-1)
    return img1,img2

# Find epilines corresponding to points in right image (second image) and
# drawing its lines on left image
lines1 = cv2.computeCorrespondEpilines(pts2.reshape(-1,1,2), 2,F)
lines1 = lines1.reshape(-1,3)
print "epilines on right image are"
print lines1
img5,img6 = drawlines(img1,img2,lines1,pts1,pts2)

# Find epilines corresponding to points in left image (first image) and
# drawing its lines on right image
lines2 = cv2.computeCorrespondEpilines(pts1.reshape(-1,1,2), 1,F)
lines2 = lines2.reshape(-1,3)
print "epilines on left image are"
print lines2
img3,img4 = drawlines(img2,img1,lines2,pts2,pts1)


#----------------------------------------#
# Epipoles from a fundamental matrix
#----------------------------------------#
"""
right_epipole = compute_epipole(F) 
left_epipole = compute_epipole(np.transpose(F))"""

#(iv) PART

epipoles = getEpipoles(F) 
right_epipole =epipoles['right']
left_epipole = epipoles['left']



ptsr[0][0]=right_epipole[0]/right_epipole[2]
ptsr[0][1]=right_epipole[1]/right_epipole[2]
ptsl[0][0]=left_epipole[0]/left_epipole[2]
ptsl[0][1]=left_epipole[1]/left_epipole[2]
# Visualizing the position of both the epipoles



color = tuple(np.random.randint(0,255,3).tolist())

img7 = cv2.circle(img5,(ptsr[0][0].astype(int),ptsr[0][1].astype(int)),10,color,-1)


#----------------------------------------#
# Epipoles from epilines
#----------------------------------------#
right_epipole1 = np.cross(lines1[0],lines1[1])
left_epipole1 = np.cross(lines2[0],lines2[1])

print "Left epipole using epilines is ", ptsr
print "Right epipole using epilines is ", ptsl

ptsr[1][0]=right_epipole1[0]/right_epipole1[2]
ptsr[1][1]=right_epipole1[1]/right_epipole1[2]
ptsl[1][0]=left_epipole1[0]/left_epipole1[2]
ptsl[1][1]=left_epipole1[1]/left_epipole1[2]

# Visualizing the position of both the epipoles

img8 = cv2.circle(img5,(ptsr[1][0].astype(int),ptsr[1][1].astype(int)),10,color,-1)
#cv2.imshow("Right Epipole using epilines",img8)
#cv2.waitKey(0)
img8 = cv2.circle(img3,(ptsl[1][0].astype(int),ptsl[1][1].astype(int)),10,color,-1)
#cv2.imshow("Left Epipole using epilines",img8)
#cv2.waitKey(0)

distr = np.linalg.norm(ptsr[0]-ptsr[1])
distl = np.linalg.norm(ptsl[0]-ptsl[1])

print "Distance between estimated right epipoles"
print distr

print "Distance between estimated left epipoles"
print distl
#----------------------------------------#
# Projection matrix estimation from F
#----------------------------------------#

#(v) PART
Te = skew(epipoles['left'])
P1 = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0]])
P2 = np.vstack((np.dot(Te,F.T).T,epipoles['left'])).T
print "P1 is assumed [I|O]"
print P1

print "P2 is computed as"
print P2


#----------------------------------------#
# Determine 3D Point
#----------------------------------------#

x = np.transpose(pts1)
x = np.vstack([x,np.ones(x.shape[1])])
print "Points from first image are"
print np.transpose(x)

y = np.transpose(pts2)
y = np.vstack([y,np.ones(y.shape[1])])
print "Points from second image are"
print np.transpose(y)

x1=np.zeros((3,8))
y1=np.zeros((3,8))

x1=np.array([[pts1[0][0],pts1[1][0],pts1[2][0],pts1[3][0],pts1[4][0],pts1[5][0],pts1[6][0],pts1[7][0]],[pts1[0][1],pts1[1][1],pts1[2][1],pts1[3][1],pts1[4][1],pts1[5][1],pts1[6][1],pts1[7][1]]])


y1=np.array([[pts2[0][0],pts2[1][0],pts2[2][0],pts2[3][0],pts2[4][0],pts2[5][0],pts2[6][0],pts2[7][0]],[pts2[0][1],pts2[1][1],pts2[2][1],pts2[3][1],pts2[4][1],pts2[5][1],pts2[6][1],pts2[7][1]]])

#(vi) PART
get_3D_coordinates(P1,P2,x1,y1)
'''
pts3d = cv2.triangulatePoints(P1, P2, x1, y1)
pts3d = np.transpose(pts3d)
print "points in 3d"
print pts3d

stereo = cv2.createStereoBM(numDisparities=16, blockSize=15)
disparity = stereo.compute(img1,img2)
plt.imshow(disparity,'gray')
plt.show()
'''
