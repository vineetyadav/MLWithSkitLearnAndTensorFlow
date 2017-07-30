from sklearn.linear_model import SGDRegressor
from numpy import np
sgd_reg = SGDRegressor()
X = 2*np.random.rand(100,1)
y = 4*3*X + np.random.rand(100,1)
sgd_reg.fit(X,y.ravel())