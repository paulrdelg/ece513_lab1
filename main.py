# Standard Python Packages
import os
import platform
import sys

# Common Third-Party Packages
import matplotlib.pyplot as plt
import numpy as np
import PIL
from sklearn.decomposition import PCA

def extractFileNames(dirName):
	onlyfiles = [f for f in os.listdir(dirName) if os.path.isfile(os.path.join(dirName, f))]
	return onlyfiles

def definePrefix(subjectNum):
	prefix = 'subject'
	if subjectNum < 10:
		prefix = prefix + '0' + str(subjectNum)
	else:
		prefix = prefix + str(subjectNum)
	return prefix

def splitSubjects(flist):
	subjects = []
	
	subjectNum = 1
	subjectFiles = []
	for f in flist:
		
		currentPrefix = definePrefix(subjectNum)
		currentSubject = f.startswith(currentPrefix)
		
		prefixSubject = 'subject'
		suffix = '.gif'
		
		if currentSubject:
			# strip file gunk
			feature = f.removeprefix(prefixSubject).removesuffix(suffix)
			#print(feature + ' is current for ' + str(subjectNum))
			
			# add feature to current subject
			subjectFiles.append(feature)
		else:
			# save previous subject
			subjects.append(subjectFiles)
			
			# define next subject
			subjectNum = subjectNum + 1
			newPrefix = definePrefix(subjectNum)
			currentSubject = f.startswith(newPrefix)
			
			# strip file gunk
			feature = f.removeprefix(prefixSubject).removesuffix(suffix)
			#print(feature + ' is new for ' + str(subjectNum))
			
			# create new subject
			subjectFiles = [feature]
	
	subjects.append(subjectFiles)
	
	return subjects

def getSubjectFeatures(subjects, subjectNumber):
	subject = subjects[subjectNumber - 1]
	return subject

def readImage(filePath):
	# read img
	img = plt.imread(filePath)
	return img

def loadData(dirPath, subjects):
	# Initialize array to be returned
	pltImagesArray = []
	
	for subject in subjects:
		for feature in subject:
			filepath = dirPath + '/subject' + feature + '.gif'
			im = readImage(filepath)
			pltImagesArray.append(im)
	
	return pltImagesArray

def main():
	print('\nRunning main script')
	yaleFacesPath = './yalefaces'
	features = ['centerlight', 'glasses', 'happy', 'leftlight', 'noglasses', 'normal', 'rightlight', 'sad', 'sleepy', 'surprised', 'wink']
	
	flist = extractFileNames(yaleFacesPath)
	
	subjects = splitSubjects(flist)
	
	pltImagesArray = loadData(yaleFacesPath, subjects)
	
	# Convert 2D images to list of 1D vectors
	N = pltImagesArray[0].shape[0] * pltImagesArray[0].shape[1]
	list_of_x_vectors = []
	for pltImage in pltImagesArray:
		x = np.asarray(pltImage).flatten()
		x = x.reshape((x.shape[0], 1))
		list_of_x_vectors.append(x)
	
	M = len(list_of_x_vectors)
	
	# Sum the x vectors
	x_sum = np.zeros(list_of_x_vectors[0].shape)
	for x in list_of_x_vectors:
		x_sum = x_sum + x
	
	print(x_sum.shape)
	
	# Divide by M scalar
	m = x_sum/M
	
	print('m:   ', m.shape)
	
	# compute mean centered image
	list_of_w_vectors = []
	WT = np.zeros((M, N))
	print('WT:  ', WT.shape)
	
	count = 0
	for x in list_of_x_vectors:
		w = x - m
		list_of_w_vectors.append(w)
		WT[count][:] = w.reshape(1, N)
		count = count + 1
	
	W = WT.transpose()
	print('W:   ', W.shape)
	print('WT:  ', WT.shape)
	
	# Compute Covariance
	C = np.matmul(W, WT, dtype=np.float32)
	
	print('C:   ', C.shape)
	
	# matrix of data (m x d)
	rows = 320
	cols = 243
	m = len(pltImagesArray)
	print(m)
	d = rows * cols
	X = np.reshape(pltImagesArray, (m, d))
	
	# Apply PCA
	U, Sigma, VT = np.linalg.svd(X, full_matrices=False)
	
	# Sanity check (dimensions)
	print('X:     ', X.shape)
	print('U:     ', U.shape)
	print('Sigma: ', Sigma.shape)
	print('VT:    ', VT.shape)
	
	num_components = 11 # Number of principal components
	Y = np.matmul(X, VT[:num_components,:].T)
	print('Y:     ', Y.shape)
	
	yImg = PIL.Image.fromarray(Y)
	#yImg.show()
	
	return 0

if __name__ == "__main__":
	if platform.python_version_tuple()[0] == 3 and platform.python_version_tuple()[1] < 9:
		print('ERROR: Need Python 3.9.X to run')
	else:
		main()
