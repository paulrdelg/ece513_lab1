# Standard Python Packages
import os
import platform
import sys

# Common Third-Party Packages
import matplotlib.pyplot as plt
import numpy as np
import PIL
from sklearn.decomposition import PCA

# Custom
import lab1

def main():
	# Load matrix of 1D column vectors (X)
	X = np.load('mydata/X.npy')
	print('X:   ', X.shape)
	
	# Retrieve Dimensions
	N = X.shape[0]
	M = X.shape[1]
	print('N:   ', N)
	print('M:   ', M)
	
	# Load SVD
	U = np.load('mydata/U.npy')
	Sigma = np.load('mydata/Sigma.npy')
	VT = np.load('mydata/VT.npy')
	print('U:   ', U.shape)
	print('S:   ', Sigma.shape)
	print('VT:  ', VT.shape)
	
	# Determine mean image (m)
	m = np.load('mydata/m.npy')
	print('m:   ', m.shape)
	
	# compute mean centered matrix (W)
	W = np.load('mydata/W.npy')
	WT = W.transpose()
	print('W:   ', W.shape)
	print('WT:  ', WT.shape)
	
	# Compute Covariance (C)
	C = np.load('mydata/C.npy')
	
	# Eigenvalues & Eigenvectors
	w = np.linalg.eigvals(C)
	np.save('mydata/eigenvalues')
	print(w)
	w2, v = np.linalg.eig(C)
	print(w2)
	print(v)
	
	return 0

if __name__ == "__main__":
	if platform.python_version_tuple()[0] == 3 and platform.python_version_tuple()[1] < 9:
		print('ERROR: Need Python 3.9.X to run')
	else:
		lab1.processed()
