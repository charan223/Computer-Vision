import sys
sys.path.append('/usr/lib/python2.7/dist-packages')
import numpy as np
import cv2
from PIL import Image
from numpy import *
from math import sqrt
import sys

if (len(sys.argv) < 2):
    print("Pass the path to input file as an argument")
    exit()
input_image_file = sys.argv[1]


#------------------------------------#
# Library implementation
#------------------------------------#

def execute_library_function():
    # Read image and convert to grayscale
    lego = cv2.imread(input_image_file)
    graylego = cv2.cvtColor(lego,cv2.COLOR_BGR2GRAY)

    # Harris Corner Detection Library Implementation

    img = lego

    gray = graylego
    gray = np.float32(gray)

    dst = cv2.cornerHarris(gray,2,3,0.04)

    # result is dilated for marking the corners, not important
    dst = cv2.dilate(dst,None)

    # Threshold for an optimal value, it may vary depending on the image.
    img[dst>0.01*dst.max()]=[0,0,255]

    cv2.imwrite("Library_harris_output.png",img)
    cv2.imshow('Library_harris_output',img)
    if cv2.waitKey(0) & 0xff == 27:
        cv2.destroyAllWindows()


if (len(sys.argv) >2):
    if (sys.argv[2]=="library"):
        execute_library_function()
    else:
        print "Improper syntax"
    exit()


#------------------------------------#
# Manual Implementation
#------------------------------------#

output_image_file = 'Harris_output.jpg'

image = Image.open(input_image_file)
imageWidth, imageHeight = image.size
imageArray = array(image)

grayScaleImage = array(image.convert('L'))

def imageGradient(image, isHorizontal):
    grad = zeros((imageHeight, imageWidth))

    for h in range(1, imageHeight - 1):
        for w in range(1, imageWidth - 1):
            if isHorizontal:
              grad[h,w] = float(int(image[h-1,w]) - int(image[h+1,w])) / 255
            else:
              grad[h,w] = float(int(image[h,w-1]) - int(image[h,w+1])) / 255

    return grad

dx = imageGradient(grayScaleImage, True)
dy = imageGradient(grayScaleImage, False)

# parameters

fragmentWidth = 5
fragmentHeight = 5
k = 0.04
z = 1e-4
maxPointsDistance = 5

gaussianCore = [[ 2,  4,  5,  4, 2],
                [ 4,  9, 12,  9, 4],
                [ 5, 12, 15, 12, 5],
                [ 4,  9, 12,  9, 4],
                [ 2,  4,  5,  4, 2]]

gaussianCoreNormalized = [[float(x) / 159 for x in list] for list in gaussianCore]

def getStructureTensor(u, v):
    mA = 0
    mB = 0
    mC = 0
    mD = 0

    for h in range(0, fragmentHeight):
        for w in range(0, fragmentWidth):
            g = gaussianCoreNormalized[h][w]
            dxi = dx[u + h - fragmentHeight / 2, v + w - fragmentWidth / 2]
            dyi = dy[u + h - fragmentHeight / 2, v + w - fragmentWidth / 2]
            mA += g * dxi * dxi
            mB += g * dxi * dyi
            mC += g * dxi * dyi
            mD += g * dyi * dyi

    return [[mA, mB], [mC, mD]]

def cornerMeasure(u, v):
    [[a, b], [c, d]] = getStructureTensor(u, v)
    det = a * d - b * c
    trace = a + d
    return det - k * (trace ** 2);

def harris():
    corners = []
    for h in range(fragmentHeight / 2, imageHeight - fragmentHeight / 2):
        for w in range(fragmentWidth / 2, imageWidth - fragmentWidth / 2):
            if (cornerMeasure(h, w) > z):
                corners.append((h, w))
    return corners

def distance((x, y), (u, v)):
    d = sqrt((x - u) ** 2 + (y - v) ** 2)
    return d

def save():
    resultImage = Image.fromarray(imageArray).save(output_image_file)
    cv2.imshow('harris_output',output_image_file)
    print "Output image is saved as", output_image_file


def drawCornersOnImage():
    corners = harris()

    for (h,w) in corners:
        imageArray[h,w] = [0,0,255]

    save()

drawCornersOnImage()
execute_library_function()
