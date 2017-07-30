from sklearn import dataset
iris = dataset.load_iris()
list(iris.keys())
X = iris["data"][:,3]