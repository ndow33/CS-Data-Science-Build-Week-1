from sklearn.datasets import load_iris
from decisionTree import DecisionTreeClassifier
from sklearn import tree

dataset = load_iris()
X, y = dataset.data, dataset.target
clf = DecisionTreeClassifier(max_depth=5)
clf.fit(X, y)
print('')
print(':::::::::::::PREDICTIONS:::::::::::::::::::::')
print('')
print(':::::::::::::::::::::::::::::::::::::::::::::')
inputs = [[1, 1.5, 5, 1.5]]
print(f'INPUTS: {inputs}')
print(f'OUR MODEL PREDICTION: {clf.predict(inputs)}')

clf2 = tree.DecisionTreeClassifier(max_depth=5)
clf2.fit(X, y)

print(f'SCIKITLEARN MODEL PREDICTION: {clf2.predict(inputs)}')
print(':::::::::::::::::::::::::::::::::::::::::::::')