
import os
import tarfile
from six.moves import urllib 
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import hashlib
from sklean.model_selection import train_test_split
from sklearn.model import StartifiedShuffleSplit
DOWNLOAD_ROOT = "https://raw.githubusercotent.com/argeron/handson-ml/master"
HOUSING_PATH= "dataset/housing"
HOUSING_URL= DOWNLOAD_ROOT+HOUSING_PATH+"/housing.tgz"

#the function is used to download housing data
def fetch_housing_data(housing_url= HOUSING_URL, housing_path=HOUSING_PATH):
    if not os.path.isdir(housing_path):
        os.makedirs(housing_path)
    tgz_path=os.path.join(housing_path,"housing.tgz")
    urllib.request.urlretrieve(housing_url,tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()

#the function is used to load housing data 
def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path,"housing.csv")
    return pd.read_csv(csv_path)

#the Function is used to split test and Train Data
def split_train_test(data,test_ratio):
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data)*test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    #iloc would return index of data 
    return data.iloc[train_indices], data.iloc[test_indices]

#The function would create hash and return same random sample data again
def test_set_check(identifier,test_ratio,hash):
    return hash(np.int64(identifier).digest()[-1]<256*test_ratio)

def split_train_test_by_id(data,test_ratio,id_column,hash=hashlib.md5):
    ids = data[id_column]
    in_test_set = ids.apply(lamba id_:test_set_check(id_,test_ratio,check))
    return data_loc[-in_test_set],data.loc[in_test_set]



housing = load_housing_data()
housing['ocean_proximity'].value_counts()
#The function is used to generate summary
housing.describe()
#hist method is called on whole dataset to generate histogram 
housing.hist(bins=50,figsize=(20,15))
plt.show()
train_set,test_set = split_train_test(housing,0.2)
print(len(train_set),"train+",len(test_set))

#adds an index column
housing_with_id = housing.reset_index()

train_set,test_set = split_train_test_by_id(housing_with_id,0.2,"index")
#create an unique identifier
housing_with_id["id"] = housing["longitude"]*1000+housing["latitude"]
train_set,test_set = split_train_test_by_id(housing_with_id,0.2,"id")

train_test,test_set = train_test_split(housing,test_size=0.2,random_state=42)

housing["income_cat"].value_counts()/len(housing)

