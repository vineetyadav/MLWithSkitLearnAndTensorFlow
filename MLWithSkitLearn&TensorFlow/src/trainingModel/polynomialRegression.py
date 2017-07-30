from sklearn.preprocessing import PolynomialFeatures 
import numpy as np
from sklearn.linear_model import LinearRegression
X = 2*np.random.rand(100,1)
y = 4*3*X + np.random.rand(100,1)
poly_features = PolynomialFeatures(degree=2,include_bais=False)
x_poly = poly_features.fit_transform(X)
print X[0]
print x_poly[0]
lin_reg = LinearRegression()
lin_reg.fit(x_poly,y)
print lin_reg.intercept_
print lin_reg.coef_

