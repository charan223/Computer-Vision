### By R.Sri Charan Reddy 14CS10037

### Instructions to Use

"Download 'lego.tif' image,it is not kept in the folder because of memory constraints"

### Gaussian noise

Copy "lego.tif" file into noise directory
Run the command -> `python noise.py`
Output images displayed are cap.bmp,lego.tif
Enter the value of s(Standard Deviation)
Median,mean PSNR values are printed and a plot containing (Original,noisy,mean,median filtered images) are displayed
Run the command -> `python plot.py` for outputting the plot "plot.png"
PSNR values are stored in "PSNR values.txt" (In this case,I got both Median and Mean PSNR values equal)
Library noise filtered images are stored


### Laplacian

Copy "lego.tif" file into laplacian directory
Run the command -> `python laplacian.py -i lego.tif` or `python laplacian.py --image lego.tif`
Laplacian and Zero crossing images are displayed
Edge pixels are printed in the file "edgepixels.txt"
Library Laplacian image is stored

### DCT

Run the command -> `python dct.py cap.bmp`
Took Threshold values for (i+j < 6) as -20
			  (5 <i+j<10) as 0
			  (i+j > 10) as 0
Block DCT transformed and Inverse transformed images are displayed and stored

### Harris corner detection

Copy "lego.tif" file into harris directory
Run the command -> `python harris.py lego.tif`
Harris and Library Harris images are displayed and stored



