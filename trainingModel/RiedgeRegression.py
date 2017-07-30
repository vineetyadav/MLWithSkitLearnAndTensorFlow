from sklearn.linear_model import Ridge
import numpy as np
X = 2*np.random.rand(100,1)
y = 4*3*X + np.random.rand(100,1)
ridge_reg = Ridge(alpha=1,solver="cholesky")
ridge_reg.fit(X,y)
ridge_reg.predict([[1.5]])
sgd_reg = SGDRgressor(penalty="l2")
sgd_reg.fit(X,y.ravel())
sgd_reg.predict([1.5])
