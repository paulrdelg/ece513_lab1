# Standard Python Packages
import os
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

def loadData(subjects):
	# Initialize array to be returned
	pltImagesArray = []
	
	for subject in subjects:
		for feature in subject:
			filepath = yaleFacesPath + '/subject' + feature + '.gif'
			im = readImage(filepath)
			pltImagesArray.append(im)
	
	return pltImagesArray

def main():
	print('\nRunning main script')
	yaleFacesPath = './yalefaces'
	flist = extractFileNames(yaleFacesPath)
	subjects = splitSubjects(flist)
	features = ['centerlight', 'glasses', 'happy', 'leftlight', 'noglasses', 'normal', 'rightlight', 'sad', 'sleepy', 'surprised', 'wink']
	return 0

if __name__ == "__main__":
	
	yaleFacesPath = './yalefaces'
	features = ['centerlight', 'glasses', 'happy', 'leftlight', 'noglasses', 'normal', 'rightlight', 'sad', 'sleepy', 'surprised', 'wink']
	
	flist = extractFileNames(yaleFacesPath)
	
	subjects = splitSubjects(flist)
	
	pltImagesArray = loadData(subjects)
	
	# Normalize
	imagesNpArray = np.asarray(pltImagesArray)
	imagesNpArray = imagesNpArray.transpose()
	
	avg_face = imagesNpArray.mean(axis=1)
	#avg_face = avg_face.reshape(imagesNpArray.shape[0], 1)
	#normalized_face = imagesNpArray - avg_face
	
	# matrix of data (m x d)
	rows = 320
	cols = 243
	m = len(pltImagesArray)
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
	yImg.show()
	
	#main()
