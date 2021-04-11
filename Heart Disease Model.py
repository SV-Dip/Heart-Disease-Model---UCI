
"Created on Sat Apr 10 08:33:37 2021"

# Imports...

from io import StringIO
import itertools
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import statistics as stats
import scipy
import pydotplus

from sklearn import preprocessing
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import mean_squared_error
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

import matplotlib.pyplot as plt
import matplotlib.gridspec as gd
import matplotlib.image as mpimg
import seaborn as sns

import tensorflow
print("TensorFlow == ", tensorflow.__version__)

import keras
print("keras == ", keras.__version__)

from keras.models import Sequential
from keras.layers import Dense

# Import Dataset
df = pd.read_csv("heart.csv")

"Data Exploration"

# Heart Disease Distribution
X1 = [round(len(df[df.target == 0])*100/len(df.target), 2), round(len(df[df.target == 1])*100/len(df.target), 2)]
Y1 = ['Without Heart Disease', 'With Heart Disease']
plot1 = sns.barplot(x = Y1, y = X1, palette = 'mako_r')
sns.set_theme(style = "white")
for bar in plot1.patches:
    plot1.annotate(format(bar.get_height(), '.2f'), (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                   ha = 'center', va = 'center', size = 12, xytext=(0, 8), textcoords='offset points')
sns.despine(left=True, bottom=True)
plt.tick_params(labelleft=False, left=False)
plt.xlabel("Heart Disease Distribution")
plt.show()

# Heatmap
sns.set_theme(style = "white")

corr = df.corr()     # Correlation matrix
mask = np.triu(np.ones_like(corr, dtype = bool))      # Generate mask for upper triangle

f, ax = plt.subplots(figsize = (20, 20))
cmap = sns.diverging_palette(250, 0, as_cmap = True)
sns.heatmap(corr, mask = mask, cmap = cmap, vmax = .3, center = 0,
            square = True, linewidths = .5, cbar_kws = {"shrink": .5})
plt.xlabel("\nHeatmap of Correlation between features")
plt.show()

"Numerical Data vs Target"

# Numerical Data Distribution
viz = df[["age", "trestbps", "chol", "thalach", "oldpeak", "target"]]
g = sns.PairGrid(viz, hue = "target")
g = g.map_diag(sns.histplot)
g = g.map_lower(sns.scatterplot)
g = g.map_upper(sns.kdeplot)

# Boxplot of Numerical features
fig = plt.figure(constrained_layout = True, figsize = (18, 12))
grid = gd.GridSpec(ncols = 4, nrows = 3, figure = fig)
sns.set_theme(style = "darkgrid")

ax1 = fig.add_subplot(grid[0, :2])
ax1.set_title('Resting Blood Pressure Distribution')
sns.boxplot(x = 'target', y = 'trestbps', data = df, palette = ['#342D7E', '#E67451'], ax = ax1)
sns.swarmplot(x = 'target', y = 'trestbps', data = df, palette = ['#E67451', '#342D7E'], ax = ax1)

ax2 = fig.add_subplot(grid[0, 2:])
ax2.set_title('Cholesterol Distribution')
sns.boxplot(x = 'target', y = 'chol', data = df, palette = ['#342D7E', '#E67451'], ax = ax2)
sns.swarmplot(x ='target', y = 'chol', data = df, palette = ['#E67451', '#342D7E'], ax = ax2)

ax3 = fig.add_subplot(grid[1, :2])
ax3.set_title('Max Heart Rate Distribution')
sns.boxplot(x = 'target', y = 'thalach', data = df, palette = ['#342D7E', '#E67451'], ax = ax3)
sns.swarmplot(x = 'target', y = 'thalach', data = df, palette = ['#E67451', '#342D7E'], ax=ax3)

ax4 = fig.add_subplot(grid[1, 2:])
ax4.set_title('ST Depression Distribution')
sns.boxplot(x = 'target', y = 'oldpeak', data = df, palette = ['#342D7E', '#E67451'], ax = ax4)
sns.swarmplot(x = 'target', y = 'oldpeak', data = df, palette = ['#E67451', '#342D7E'], ax = ax4)

ax5 = fig.add_subplot(grid[2, :])
ax5.set_title('Age Distribution')
sns.boxplot(x = 'target', y = 'age', data = df, palette = ['#342D7E', '#E67451'], ax = ax5)
sns.swarmplot(x = 'target', y = 'age', data = df, palette = ['#E67451', '#342D7E'], ax = ax5)

plt.show()

# Age Distribution
fig = plt.figure(constrained_layout = True, figsize = (12, 4))
sns.set_theme(style = "white")
sns.histplot(x = 'age', data = df, hue = 'target',bins = 40, multiple = 'dodge', palette = 'bwr', edgecolor = ".3")
sns.despine(left = True)
plt.tick_params(labelleft=False, left=False)
plt.legend(["With Disease", "Without Disease"])
plt.ylabel(" ")
plt.xlabel("Heart Disease Frequency for Ages", fontsize = 13)
plt.show()

# Age and Max Heart Rate vs Target
fig = plt.figure(constrained_layout = True, figsize = (12, 5))
grid = gd.GridSpec(ncols = 2, nrows = 1, figure = fig)
sns.set_theme(style = "dark")

ax1 = fig.add_subplot(grid[0, :1])
sns.despine(fig, left = True, bottom = True, ax = ax1)
sns.scatterplot(x = "age", y = "thalach", hue = "target", linewidth = 0, sizes = (1, 8), data = df, ax = ax1)
plt.legend(["With Disease", "Without Disease"])
plt.xlabel("Age", fontsize = 12)
plt.ylabel("Maximum Heart Rate", fontsize = 12)
plt.tick_params(labelleft = False, left = False)
plt.tick_params(labelbottom = False, bottom = False)

ax2 = fig.add_subplot(grid[0, 1:])
sns.despine(fig, left = True, bottom = True)
sns.kdeplot(x = 'age', y = 'thalach', hue = 'target', data = df, ax = ax2)
plt.xlabel("Age", fontsize = 12)
plt.ylabel("Maximum Heart Rate", fontsize = 12)
plt.tick_params(labelleft = False, left = False)
plt.tick_params(labelbottom = False, bottom = False)

plt.show()

"Categorical Data vs Target"

def plotcategoricaldata(X, Y, m, n, text):
    fig = plt.figure(constrained_layout = True, figsize = (3 + n, 5))
    grid = gd.GridSpec(ncols = 2, nrows = 1, figure = fig)
    sns.set_theme(style = "dark")
    
    ax1 = fig.add_subplot(grid[0, :1])
    sns.despine(fig, left = True, bottom = True, ax = ax1)
    plot = sns.barplot(x = Y, y = X, palette = 'mako_r')
    for bar in plot.patches:
        plot.annotate(format(bar.get_height(), '.2f'), (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                        ha = 'center', va = 'center', size = 12, xytext = (0, 8), textcoords = 'offset points')
    sns.despine(left = True, bottom = True)
    plt.tick_params(labelleft = False, left = False)
    plt.ylim([0, 100])
    
    ax2 = fig.add_subplot(grid[0, 1:])
    sns.despine(fig, left = True, bottom = True, ax = ax2)
    sns.histplot(x = m, data = df, hue = 'target', bins = n, multiple = 'fill', palette = 'mako_r', ax = ax2)
    plt.tick_params(labelleft = False, left = False)
    plt.tick_params(labelbottom = False, bottom = False)
    plt.legend(["With Disease", "Without Disease"])
    plt.ylabel(" ")
    plt.xlabel(text, fontsize = 13)
    
    plt.show()

# Sex Distribution
X = [round(len(df[df.sex == 0])*100/len(df.sex), 2),
     round(len(df[df.sex == 1])*100/len(df.sex), 2)]
Y = ['female', 'male']
m = 'sex'
n = 3
text = 'female                              male\nHeart Disease Frequency for Sex'
plotcategoricaldata(X, Y, m, n, text)

# Chest Pain Type Distribution
X = [round(len(df[df.cp == 0])*100/len(df.cp), 2),
     round(len(df[df.cp == 1])*100/len(df.cp), 2),
     round(len(df[df.cp == 2])*100/len(df.cp), 2),
     round(len(df[df.cp == 3])*100/len(df.cp), 2)]
Y = ['Type 0', 'Type 1', 'Type 2', 'Type 3']
m = 'cp'
n = 7
text = 'Type 0               Type 1                Type 2                 Type 3\nHeart Disease Frequency for Chest Pain Type'
plotcategoricaldata(X, Y, m, n, text)

# Fasting Blood Sugar Distribution
X = [round(len(df[df.fbs == 0])*100/len(df.fbs), 2),
     round(len(df[df.fbs == 1])*100/len(df.fbs), 2)]
Y = ['False', 'True']
m = 'fbs'
n = 3
text = 'False               True\nHeart Disease Frequency for Fasting Blood Sugar\nfbs > 120mg/dl (0 if False, 1 if True)'
plotcategoricaldata(X, Y, m, n, text)

# Resting Electrocardiographic Distribution
X = [round(len(df[df.restecg == 0])*100/len(df.restecg), 2),
     round(len(df[df.restecg == 1])*100/len(df.restecg), 2),
     round(len(df[df.restecg == 2])*100/len(df.restecg), 2)]
Y = ['0', '1', '2']
m = 'restecg'
n = 5
text = '0                          1                           2\nHeart Disease Frequency for Rest ECG'
plotcategoricaldata(X, Y, m, n, text)

# Exercise Induced Angina Distribution
X = [round(len(df[df.exang == 0])*100/len(df.exang), 2),
     round(len(df[df.exang == 1])*100/len(df.exang), 2)]
Y = ['0', '1']
m = 'exang'
n = 3
text = '0                                    1\nFrequency for Exercise Induced Angina'
plotcategoricaldata(X, Y, m, n, text)

# Slope of peak ST segment DIstribution
X = [round(len(df[df.slope == 0])*100/len(df.slope), 2),
     round(len(df[df.slope == 1])*100/len(df.slope), 2),
     round(len(df[df.slope == 2])*100/len(df.slope), 2)]
Y = ['0', '1', '2']
m = 'slope'
n = 5
text = '0                      1                      2\nHeart Disease Frequency for Slope of Peak ST Segment'
plotcategoricaldata(X, Y, m, n, text)

# Number of Major Vessels Distribution
X = [round(len(df[df.ca == 0])*100/len(df.ca), 2),
     round(len(df[df.ca == 1])*100/len(df.ca), 2),
     round(len(df[df.ca == 2])*100/len(df.ca), 2),
     round(len(df[df.ca == 3])*100/len(df.ca), 2),
     round(len(df[df.ca == 4])*100/len(df.ca), 2)]
Y = ['0', '1', '2', '3', '4']
m = 'ca'
n = 9
text = '0                      1                       2                       3                       4\nHeart Disease Frequency for No. of Major Vessels'
plotcategoricaldata(X, Y, m, n, text)

print("\nModel Building\n")

print("Preprocessing")
y = df.target.values
X = df.drop(['target'], axis = 1)

X = preprocessing.StandardScaler().fit(X).transform(X)

x_train, x_test, y_train, y_test = train_test_split( X, y, test_size = 0.2, random_state = 4)

print ('Train set:', x_train.shape,  y_train.shape)
print ('Test set:', x_test.shape,  y_test.shape)

print("\nLogistic Regression\n")

print("Fit LR Object")
LR = LogisticRegression(C = 0.01, solver = 'liblinear').fit(x_train,y_train) # Fit LR Object
print(LR)
yhat_LR = LR.predict(x_test)  # Model Predict

print("\nEvaluation of LR Model")
y_1 = mean_squared_error(y_test, yhat_LR)
print("Mean Squared Error of Logistic Regression Model = ", y_1, "\n")

# Confusion Matrix
def plot_confusion_matrix(cm, classes, normalize = False, title = 'Confusion matrix', cmap = plt.cm.Blues):
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis = 1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation = 'nearest', cmap = cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation = 45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment = "center",
                 color = "white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# Compute confusion matrix
cnf_matrix_LR = confusion_matrix(y_test, yhat_LR, labels = [1,0])
np.set_printoptions(precision = 2)

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix_LR, classes = ['heart disease', 'without heart disease'],
                      normalize = False,  title = 'Confusion matrix for Logistic Regression')

print (classification_report(y_test, yhat_LR))

print("\nDecision Tree\n")

print("Fit Decision Tree Object")
Tree = DecisionTreeClassifier(criterion="entropy", max_depth = 4)
print(Tree)
Tree.fit(x_train, y_train)    # Fit Decision Tree Object
predTree = Tree.predict(x_test)    # Model Predict

print("\nEvaluation of Decision Tree Model")
y_2 = mean_squared_error(y_test, predTree)
print("Mean Squared Error of Decision Tree Model = ", y_2, "\n")

# Compute confusion matrix
cnf_matrix_tree = confusion_matrix(y_test, predTree, labels = [1,0])
np.set_printoptions(precision = 2)

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix_tree, classes = ['heart disease', 'without heart disease'],
                      normalize = False,  title = 'Confusion matrix for Decision Tree')

print (classification_report(y_test, predTree))

# Visualization
dot_data = StringIO()
filename = "tree.png"
featureNames = df.columns[0:13]
targetNames = df["target"].unique().tolist()

out = tree.export_graphviz(Tree,feature_names = featureNames, out_file = dot_data,
                           class_names =  (["With Disease", "Without Disease"]), filled = True,
                           special_characters = True, rotate = False)

graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_png(filename)
img = mpimg.imread(filename)
plt.figure(figsize = (200, 400))
plt.imshow(img, interpolation = 'nearest')
plt.show()

print("\n\nNeural Network Model\n")

model = Sequential()
model.add(Dense(10, activation = 'relu', input_shape = (x_train.shape[1],)))
model.add(Dense(10, activation = 'relu'))
model.add(Dense(1, activation = 'sigmoid'))

model.compile(optimizer = 'adam', loss = 'mean_squared_error', metrics = ["mean_squared_error"])

model.summary()

history = model.fit(x_train, y_train, validation_split = 0.3, epochs = 150)

print("\nModel Evaluation\n")
f1_train = f1_score(y_train, np.around(model.predict(x_train)))
f1_test = f1_score(y_test, np.around(model.predict(x_test)))

print("F1 score on trainset = ", f1_train)
print("F1 score on testset = ", f1_test)

plt.plot(history.history['mean_squared_error'])
sns.despine(left = True, bottom = True)
sns.set_theme(style = "dark")
plt.tick_params(labelleft=False, left=False)
plt.title('Root Mean Squared Error', fontsize = 14)
plt.ylabel('rmse')
plt.xlabel('epoch')
plt.ylabel(' ')
plt.show()

print("\nTest Accuracy\n")
mse_train = mean_squared_error(y_train, model.predict(x_train))
mse_test = mean_squared_error(y_test, model.predict(x_test))

print("Training Set Accuracy = ", 100 - mse_train*100)
print("Testing Set Accuracy = ", 100 - mse_test*100, "\n")

# Convert test data
y_pred_test = np.around(model.predict(x_test))

# Compute confusion matrix
cnf_matrix_nn = confusion_matrix(y_test, y_pred_test, labels = [1,0])
np.set_printoptions(precision = 2)

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix_nn, classes = ['heart disease', 'without heart disease'],
                      normalize = False,  title = 'Confusion matrix for Neural Network')

print (classification_report(y_test, y_pred_test))

# Learning Curves

history_dict = history.history
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']
accuracy_ = history_dict['mean_squared_error']
val_accuracy_ = history_dict['val_mean_squared_error']

accuracy = []
val_accuracy = []
for i in range(len(accuracy_)):
    accuracy.append(1 - accuracy_[i])
    val_accuracy.append(1 - val_accuracy_[i])

epochs = range(1, len(loss_values) + 1)

# Plot model accuracy-vs-epochs and loss-vs-epoch
fig, ax = plt.subplots(1, 2, figsize = (15, 6))
sns.set_theme(style = "dark")
x_dot = [0, 150]
y_dot_accuracy = [max(val_accuracy), max(val_accuracy)]
y_dot_loss = [min(val_loss_values), min(val_loss_values)]

ax[0].plot(epochs, accuracy, 'r', label = 'Training accuracy', linewidth = 2)
ax[0].plot(epochs, val_accuracy, 'b', label = 'Validation accuracy', linewidth = 2)
ax[0].plot(x_dot, y_dot_accuracy, linestyle = (0,(0.1,2)), dash_capstyle = 'round', linewidth = 2, color = 'Black')
ax[0].set_title('Training - Validation Accuracy', fontsize=12)
ax[0].set_xlabel('Epochs', fontsize = 10)
ax[0].set_ylabel('Accuracy', fontsize = 10)
ax[0].spines['right'].set_visible(False)
ax[0].spines['top'].set_visible(False)
ax[0].spines['left'].set_visible(False)
ax[0].legend()

ax[1].plot(epochs, loss_values, 'r', label = 'Training loss', linewidth = 2) 
ax[1].plot(epochs, val_loss_values, 'b', label = 'Validation loss', linewidth = 2)
ax[1].plot(x_dot, y_dot_loss, linestyle = (0,(0.1,2)), dash_capstyle = 'round', linewidth = 2, color = 'Black')
ax[1].set_title('Training - Validation Loss', fontsize = 12)
ax[1].set_xlabel('Epochs', fontsize = 10)
ax[1].set_ylabel('Loss', fontsize = 10)
ax[1].spines['right'].set_visible(False)
ax[1].spines['top'].set_visible(False)
ax[1].spines['left'].set_visible(False)
ax[1].legend()

plt.show()

model.save('heart_disease_trained.h5')
print("\n...model saved")

print("\nModel Training With 50 Models\n")

def regression_model():
    model_m = Sequential()
    model_m.add(Dense(10, activation = 'relu', input_shape = (x_train.shape[1],)))
    model_m.add(Dense(10, activation = 'relu'))
    model_m.add(Dense(1, activation = 'sigmoid'))

    model_m.compile(optimizer = 'adam', loss = 'mean_squared_error', metrics = ["mean_squared_error"])
    
    return model_m

mse_list_train = []
mse_list_test = []

for i in range(50):
    x_train, x_test, y_train, y_test = train_test_split( X, y, test_size = 0.2, random_state = 4)
    
    # Train and fit model
    model_m = regression_model()
    print('\n\n\nTraining Model # ' , i+1 , '\n\n')
    model_m.fit(x_train, y_train, validation_split = 0.3, epochs = 150)
    print('\n')
    
    # Prediction and evaluation
    mse_train = mean_squared_error(y_train, model_m.predict(x_train))
    mse_test = mean_squared_error(y_test, model_m.predict(x_test))
    print('\nMSE on Train Set for Training Model #', i+1 , ' = ', mse_train)
    print('\nMSE on Test Set for Training Model #', i+1 , ' = ', mse_test)

    print("\n", classification_report(y_test, np.around(model_m.predict(x_test))))
    
    # Append mse to mse_list
    mse_list_train.append(mse_train)
    mse_list_test.append(mse_test)

# Calculate Mean of the MSE
model_m.summary()

print("\n\nModel Evaluation on 50 models with 150 epochs")

mean_train = stats.mean(mse_list_train)
mean_test = stats.mean(mse_list_test)

print('\n\nTraining Set:')
print('Mean MSE of training set of 50 Models : ' , mean_train)
print('Standard Deviation of MSE of training set of 50 Models : ' , stats.stdev(mse_list_train))
print("Mean accuracy of training-set of 50 models = ", 100 - mean_train*100)

print('\n\nTesting Set:')
print('Mean MSE of testing set of 50 Models : ' , mean_test)
print('Standard Deviation of MSE on testing set of 50 Models : ' , stats.stdev(mse_list_test))
print("Mean accuracy of testing-set of 50 models = ", 100 - mean_test*100)

x_axis = ['Logistic Regression', 'Decision Tree', 'Keras Model\n(average over 50 models)']
y_axis = [100 - 100*y_1, 100 - 100*y_2, 100 - 100*mean_test]

plots = sns.barplot(x = x_axis, y = y_axis, palette = "rocket")
for bar in plots.patches:
    plots.annotate(format(bar.get_height(), '.2f'), (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                   ha = 'center', va = 'center', size = 12, xytext=(0, 8), textcoords='offset points')
sns.despine(left = True, bottom = True)
sns.set_theme(style = "white")
plt.tick_params(labelleft=False, left=False)
plt.ylim([0, 100])
plt.xlabel("\nModel Comparison", fontsize = 14)
plt.show()


