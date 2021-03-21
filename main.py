#importing the dataset
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import imread

def readMain(num, other):
	# directory
	fn = 'yalefaces/subject'
	
	if num < 1:
		print('Error')
		exit()
	elif num < 10:
		fn = fn + '0' + str(num) + other + '.gif'
	else:
		fn = fn + str(num) + other + '.gif'
	
	# read img
	img = imread(fn)
	return img

img = readMain(2, 'rightlight')
#img = img.astype(np.uint8)
plt.imshow(img, cmap='gray')
