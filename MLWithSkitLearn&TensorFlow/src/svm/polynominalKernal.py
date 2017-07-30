from sklearn.svm import SVC
from sklearn import datasets
from sklearn.pipeline import Pipeline 
from sklearn.preprocessing import StandardScaler
from sklearn import datasets
import numpy as np
iris = datasets.load_iris()
X = iris["data"][:(2,3)] #petal Leanth,petal Width 
y = (iris["target"]==2).astype(np.float64)
poly_kernal_svm_clf = Pipeline(("scaler",StandardScaler(),("svm_clf",SVC(kernal="poly",degree=3,coef0=1,C=5))))
poly_kernal_svm_clf.fit(X,y)
poly_kernal_svm_clf.predict([5.5,1.7])