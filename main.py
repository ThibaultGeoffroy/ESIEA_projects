import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import metrics
from sklearn.metrics import confusion_matrix
import itertools

data = pd.read_csv('data/iris_dataset')

# X2D = data[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]

X2D = np.array(data[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']])
# plt.plot(X2D[0])
# plt.plot(X2D[1])
# plt.show()

X2D = preprocessing.normalize(X2D)

y = np.array(data[['species']])
y = np.ravel(y)
x_train, x_test, y_train, y_test = train_test_split(X2D, y, test_size=0.3)

clf = svm.SVC(kernel='linear', C=1000)

clf.fit(x_train, y_train)

y_pred_train = clf.predict(x_train)
print("Accuracy on training :", metrics.accuracy_score(y_train, y_pred_train))

y_pred_test = clf.predict(x_test)

print("Accuracy on test:", metrics.accuracy_score(y_test, y_pred_test))

cf_m = confusion_matrix(y_test, y_pred_test)


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.show()


# plot_confusion_matrix(cf_m, ['setosa', 'virginica', 'versicolor'])
result_train = []
result_test = []
for i in range(1, 30):
    clf = svm.SVC( C=i)
    clf.fit(x_train, y_train)
    result_train.append(metrics.accuracy_score( y_train, clf.predict(x_train)))
    result_test.append( metrics.accuracy_score(y_test, clf.predict(x_test)))


plt.plot(result_train)
plt.plot(result_test)
plt.show()