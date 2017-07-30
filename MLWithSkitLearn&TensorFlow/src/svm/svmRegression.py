from sklearn.svm import LinearSVR
from sklearn.svm import SVR
from sklearn import datasets
import numpy as np
iris = datasets.load_iris()
X = iris["data"][:,(2,3)] #petalLentth, petalWidth
y = (iris["target"]==2).astype(np.float64)
svm_reg = LinearSVR(epsilon=1.5)
svm_reg.fit(X,y)
svm_poly_reg = SVR(kernel="poly",degree=2,C=100,epsilon=0.1)
