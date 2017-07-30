from sklearn.datasets import make_moons
from sklearn.pipeline import Pipeline
import numpy as np
from sklearn import datasets 
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
iris = datasets.load_iris()
X = iris["data"][:,(2,3)]
y = (iris["target"]==2).astype(np.float64)
polynominal_svm_clf = Pipeline((("poly_features",PolynomialFeatures(degree=3)),
                               ("scaler",StandardScaler()),
                               ("svm_clf",LinearSVC(C=10,loss="hinge"))))
polynominal_svm_clf.fit(X,y)
polynominal_svm_clf.predict([[5.5,1.7]])
