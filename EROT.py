
# Entropic regularized optimal transport

import scipy.io as sio
from sklearn.utils import shuffle

 # 1. Data

# importing the source data  
data1 = sio.loadmat('webcam.mat') 
X_s=data1['fts']
y_s=data1['labels']

#y_s=y_s.T #to make it vector
X_s, y_s = shuffle(X_s, y_s)


# importing the traget data
data2 = sio.loadmat('dslr.mat') 
X_t=data2['fts']
y_t=data2['labels']

#y_t=y_t.T #to make  it vector (transpose)
X_t, y_t = shuffle(X_t, y_t)


 # 2. Normalizing

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_s = sc.fit_transform(X_s)
X_t = sc.fit_transform(X_t)

import numpy as np

# 3 ------Subspace alignmnet alg0---------------------------------------------#

# computing d PC
from sklearn.decomposition import PCA
Xs = PCA(n_components=295).fit(X_s).components_.T  
Xt = PCA(n_components=157).fit(X_t).components_.T

#  compute alignment matrix M & Xa
M = np.dot(Xs.T, Xt)  
Xa = np.dot(Xs, M) 

# project source into aligned space
Sa = np.dot(X_s, Xa)
Ta = np.dot(X_t, Xt)  

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

clf1 = KNeighborsClassifier(n_neighbors=1)
clf1.fit(Sa, y_s.ravel())

y_pred1 = clf1.predict(Ta) 
score1 = accuracy_score(y_t.ravel(), y_pred1)

# 87.89§ accuracy 

#---------------------------------------------------------------------------#

 # 4  EROT 

# two vectors representing uniform dist on source and target data points
a = np.ones(len(X_s)) / len(X_s)
b = np.ones(len(X_t)) / len(X_t)

from scipy.spatial.distance import cdist
# loss matrix M
M2 = cdist(X_s, X_t, metric='sqeuclidean')
# normalize M by maximum value
M2 /= M2.max()

from ot import sinkhorn
# use sinnkhorn 
G = sinkhorn(a, b, M2, .001)

#value of entropic regularized
# for 1 = 76.43%
# for 10 = 76.43%
# for 100 = 76.43%
# for .1 = 74.5%
# for .01 = 85.35%
# for .001 = 81.5%

# transport/align source points
S_a = np.dot(G, X_t)


# fit 1-NN classifier
clf2 = KNeighborsClassifier(n_neighbors=1)
clf2.fit(S_a, y_s.ravel())
y_pred2 = clf2.predict(X_t)

score2 = accuracy_score(y_t.ravel(), y_pred2) 

