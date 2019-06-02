# Check the versions of libraries

# Python version
import sklearn
import pandas
import matplotlib
import numpy
import scipy
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn import model_selection
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix


import sys
print('Python: {}'.format(sys.version))
# scipy
print('scipy: {}'.format(scipy.__version__))
# numpy
print('numpy: {}'.format(numpy.__version__))
# matplotlib
print('matplotlib: {}'.format(matplotlib.__version__))
# pandas
print('pandas: {}'.format(pandas.__version__))
# scikit-learn
print('sklearn: {}'.format(sklearn.__version__))


# Load libraries
# Load dataset
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = pandas.read_csv(url, names=names)

# shape
print(dataset.shape)
