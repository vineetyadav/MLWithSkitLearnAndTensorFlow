from sklearn.linear_model import ElasticNet
import numpy as np
X = 2*np.random.rand(100,1)
y = 4*3*X + np.random.rand(100,1)
elasticNet = ElasticNet(alpha=0.1,l1_ratio=0.5)
elasticNet.fit(X,y)
elasticNet.predict([1.5])