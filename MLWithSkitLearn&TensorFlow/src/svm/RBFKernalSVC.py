import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn import datasets
from sklearn.svm import SVC
iris = datasets.load_iris()
X = iris["data"][:,(2,3)] #petalLenth, petalWidth
y = (iris["target"]==2).astype(np.float64)
rbf_kernal = Pipeline((("scaler",StandardScaler()),("svm_clf",SVC(kernel="rbf",gamma=5,C=0.001))))
rbf_kernal.fit(X,y)
