from sklean.model_selection import train_test_split
from sklearn.model import StartifiedShuffleSplit
from loadData import load_housing_data
import matplotlib.pyplot as plt
housing = load_housing_data()
split = StartifiedShuffleSplit(n_split=1,test_size=0.2,random_state=42)
for train_index,test_index in split.split(housing,housing["income_cat"]):
    strat_train_set = housing.loc[train_index]
    start_test_set = housing.loc[test_index]
housing = strat_train_set.copy()
#ploting scatter graphs
housing.plot(kind="scatter",x="longitude",y="latitude")
##ploting scatter graph with better visualization
housing.plot(kind="scatter",x="longitude",y="latitude",alpha=0.1)
#plotting california price 
housing.plot(kind="scatter",x="longitude",y="latitude",alpha=0.4, s=housing["population"]/100,label="population",c="median_housing_value"
             ,cmp =plt.get_cmap("jet"),colorbar=True)
housing["rooms_per_household"] = housing["total_rooms"]/housing["households"]
housing["bedrooms_per_room"] = housing["total_bedrooms"]/housing["total_rooms"]
housing["populations_per_household"] = housing["populations"]/housing["households"]


plt.legend()
corr_matrix = housing.corr()
corr_matrix["median_house_value"].sort_values(ascending=False)

