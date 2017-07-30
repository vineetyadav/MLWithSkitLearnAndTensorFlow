from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.linear_model import LinearRegression 
def plot_learning_curves(model,X,y):
    X_train,X_val,y_train,y_val = train_test_split(x,y,test_size=0.2)
    for m in range(1,len(X_train)):
            model.fit(X_train[:m],y_train[:m])
            y_train_predict = model.predict(X_train[:m])
            y_val_predict = model.predict(X_val)
            train_errors.append(mean_squared_error(y_train_predict,y_train[:m]))
            val_errors.append(mean_squared_error(y_val_predict,y_val))
    plt.plot(np.sqrt(train_errors),'r-+',linewidth=2,label="train")
    plt.plot(np.sqrt(val_errors),"b--",linewidth=3,labels="val")    
