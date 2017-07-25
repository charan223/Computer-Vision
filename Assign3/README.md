## Instructions to use

Mode can be:
* sift
* manual

### Sift mode:

`python fundamental.py Sunrise_Lt.jpg Sunrise_Rt.jpg sift`

The corresponding points are automatically selected using SIFT.

### Manual mode:

`python fundamental.py Sunrise_Lt.jpg Sunrise_Rt.jpg manual`

* The two images will be displayed.
* Select the corresponding points in both the images in the same order. The index will be displayed on the image.
* We need to select at least 8 points.
* After selecting the required number of corresponding points press ESC to proceed.

* The required computations will be done and then both the images with epipoles and epilines will be displayed.
* The fundamental matrix, 3d coordinates, projection matrix and calibration matrix will be logged on console.

#depth values corroborate with relative 3D positions of those points as observed in images ,this can shown from plots displayed.
