import numpy as np
from sklearn.linear_model import LinearRegression 
lin_reg = LinearRegression()
X = 2*np.random.rand(100,1)
y = 4*3*X + np.random.rand(100,1)
lin_reg.fit(X,y)
print lin_reg.intercept_
print lin_reg.coef_
print lin_reg.predict(X_new)
