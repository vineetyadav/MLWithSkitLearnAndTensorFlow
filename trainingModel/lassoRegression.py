from sklearn.linear_model import Lasso
import numpy as np
X = 2*np.random.rand(100,1)
y = 4*3*X + np.random.rand(100,1)
lasso_reg = Lasso(alpha=0.1)
lasso_reg.fit(X,y)
lasso_reg.predict([[1.5]])