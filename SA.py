

# Subspace alignment

import scipy.io as sio
from sklearn.utils import shuffle

# importing the source data  
data1 = sio.loadmat('webcam.mat') 
X_s=data1['fts']
y_s=data1['labels']

y_s=y_s.T #to make it vector
X_s, y_s = shuffle(X_s, y_s)


# importing the traget data
data2 = sio.loadmat('dslr.mat') 
X_t=data2['fts']
y_t=data2['labels']

y_t=y_t.T #to make  it vector (transpose)
X_t, y_t = shuffle(X_t, y_t)



# normalizing
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_s = sc.fit_transform(X_s)
X_t = sc.fit_transform(X_t)



# The method aims to project the labeled source and unlabeled target samples S and T 
# in two subspaces spanned by their principal components
import numpy as np

# ------Subspace alignmnet alg0---------------------------------------------#

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

#---------------------------------------------------------------------------#


#testing it with 1-NN, with without Subspace alignment algo then applying SA algo
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

#First make the predictions using raw data on 1-NN
clf1 = KNeighborsClassifier(n_neighbors=1)
clf1.fit(X_s, y_s)

y_pred1 = clf1.predict(X_t)

score1 = accuracy_score(y_t, y_pred1) 


           # score1 accuracy without SA algo is 30.57%



# fit 1-NN classifier on S_a and make predictions on T_a
clf2 = KNeighborsClassifier(n_neighbors=1)
clf2.fit(Sa, y_s)

y_pred2 = clf2.predict(Ta) 
score2 = accuracy_score(y_t, y_pred2)



          # score 2 accuracy with SA algo is 87.89%