# Load libraries
import numpy as np
from pandas import read_csv
from pandas.plotting import scatter_matrix
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import ADASYN
from imblearn.under_sampling import RandomUnderSampler
from imblearn.under_sampling import ClusterCentroids
from imblearn.under_sampling import TomekLinks

# Load dataset
url = "ImbalancedIris.csv"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = read_csv(url, names=names)
# Split-out validation dataset
array = dataset.values
X = array[:,0:4]
y = array[:,4]
X_train1, X_validation1, Y_train1, Y_validation1 = train_test_split(X, y, test_size=0.50, random_state=1, shuffle=True)
X_train2 = X_validation1
X_validation2 = X_train1
Y_train2 = Y_validation1
Y_validation2 = Y_train1


print("\n===============Part 1: Imbalanced Data Set===============\n")

# evaluate each model in turn
NNmodels1 = MLPClassifier(alpha=1, max_iter=1000)
NNmodels1.fit(X_train1, Y_train1)
predictions_NN1 = NNmodels1.predict(X_validation1)

NNmodels2 = MLPClassifier(alpha=1, max_iter=1000)
NNmodels2.fit(X_train2, Y_train2)
predictions_NN2 = NNmodels2.predict(X_validation2)

predictions_NN = np.concatenate((predictions_NN1, predictions_NN2))
test_NN = np.concatenate((Y_validation1, Y_validation2))
cm = confusion_matrix(test_NN, predictions_NN)

def getMin(a, b):
    if(a > b):
        return b
    else:
        return a

def getClassBalancedAccuracy(array):
    score = []
    sum = 0
    lenth = len(array)
    for i in range(lenth):
        sumCol = 0
        sumRow = 0
        for j in range(lenth):
            sumCol = sumCol + array[i][j]
            sumRow = sumRow + array[j][i]
        
        Precision = array[i][i]/sumCol
        Recall = array[i][i]/sumRow
        score.append(getMin(Precision, Recall))
    
    for i in range(lenth):
        sum = sum + score[i]
    
    return sum/lenth

def getBalancedAccuracy(array):
    score = []
    sum = 0
    lenth = len(array)
    for i in range(lenth):
        TPFP = 0
        TPFN = 0
        TN = 0
        TP = array[i][i]
        for j in range(lenth):
            TPFP = TPFP + array[i][j]
            TPFN = TPFN + array[j][i]
            for k in range(lenth):
                if( j!= i and k != i):
                    TN = TN + array[j][k]
        FP = TPFP - TP
        sensitivity = TP/TPFN
        specificity = TN/(TN+FP)
        score.append((sensitivity+specificity)/2)
    
    for i in range(lenth):
        sum = sum + score[i]

    return sum/lenth

#print the result for Imbalanced Data set
print("Confusion Matrix: ")
print(cm)
print("\na.Accuracy:", accuracy_score(test_NN, predictions_NN))
print("b.Class Balanced Accuracy (slide 16):", getClassBalancedAccuracy(cm))
print("c.Balanced Accuracy (slide 16):", getBalancedAccuracy(cm))
print("d.Balanced Accuracy (sklearn function):", balanced_accuracy_score(test_NN, predictions_NN))

# test = [[4, 6, 3], [1, 2, 0], [1, 2, 6]]
# print("Class Balanced test:", getClassBalancedAccuracy(test))
# print("Balanced test:", getBalancedAccuracy(test))

print("\n===============Part 2: Oversampling===============\n")

def oversampling(model):
    if (model == "random"):
        ros = RandomOverSampler(random_state=0)
        X_resampled, y_resampled = ros.fit_resample(X, y)
    elif (model == "SMOTE"):
        X_resampled, y_resampled = SMOTE().fit_resample(X, y)
    elif (model == "ADASYN"):
        try:
            X_resampled, y_resampled = ADASYN().fit_resample(X, y)
        except RuntimeError as err:
            print('Handling run-time error:', err)
            print('\nUsing SMOTE instead:')
            X_resampled, y_resampled = SMOTE().fit_resample(X, y)
        
    XR_train1, XR_validation1, YR_train1, YR_validation1 = train_test_split(
        X_resampled, y_resampled, test_size=0.50, random_state=1, shuffle=True)
    XR_train2 = XR_validation1
    XR_validation2 = XR_train1
    YR_train2 = YR_validation1
    YR_validation2 = YR_train1

    # evaluate each model in turn
    NNmodels1 = MLPClassifier(alpha=1, max_iter=1000)
    NNmodels1.fit(XR_train1, YR_train1)
    predictions_NN1 = NNmodels1.predict(XR_validation1)

    NNmodels2 = MLPClassifier(alpha=1, max_iter=1000)
    NNmodels2.fit(XR_train2, YR_train2)
    predictions_NN2 = NNmodels2.predict(XR_validation2)

    predictions_NN = np.concatenate((predictions_NN1, predictions_NN2))
    test_NN = np.concatenate((YR_validation1, YR_validation2))
    cm = confusion_matrix(test_NN, predictions_NN)
    return predictions_NN, test_NN, cm

# print the result for Oversampling
print("a. Random Oversampling:")
predictions_RO, test_RO, cm_RO = oversampling("random")
print("Confusion Matrix: ")
print(cm_RO)
print("Accuracy:", accuracy_score(test_RO, predictions_RO))

print("\nb. SMOTE Oversampling:")
predictions_SMOTE, test_SMOTE, cm_SMOTE = oversampling("SMOTE")
print("Confusion Matrix: ")
print(cm_SMOTE)
print("Accuracy:", accuracy_score(test_SMOTE, predictions_SMOTE))

print("\nc. ADASYN Oversampling:")
predictions_ADASYN, test_ADASYN, cm_ADASYN = oversampling("ADASYN")
print("Confusion Matrix: ")
print(cm_ADASYN)
print("Accuracy:", accuracy_score(test_ADASYN, predictions_ADASYN))

print("\n===============Part 3: Undersampling===============\n")

def undersampling(model):
    if (model == "random"):
        rus = RandomUnderSampler(random_state=0)
        X_resampled, y_resampled = rus.fit_resample(X, y)
    elif (model == "Cluster"):
        cc = ClusterCentroids(random_state=0)
        X_resampled, y_resampled = cc.fit_resample(X, y)
    elif (model == "Tomek"):
        X_resampled, y_resampled = TomekLinks().fit_resample(X, y)
    XR_train1, XR_validation1, YR_train1, YR_validation1 = train_test_split(
        X_resampled, y_resampled, test_size=0.50, random_state=1, shuffle=True)
    XR_train2 = XR_validation1
    XR_validation2 = XR_train1
    YR_train2 = YR_validation1
    YR_validation2 = YR_train1

    # evaluate each model in turn
    NNmodels1 = MLPClassifier(alpha=1, max_iter=1000)
    NNmodels1.fit(XR_train1, YR_train1)
    predictions_NN1 = NNmodels1.predict(XR_validation1)

    NNmodels2 = MLPClassifier(alpha=1, max_iter=1000)
    NNmodels2.fit(XR_train2, YR_train2)
    predictions_NN2 = NNmodels2.predict(XR_validation2)

    predictions_NN = np.concatenate((predictions_NN1, predictions_NN2))
    test_NN = np.concatenate((YR_validation1, YR_validation2))
    cm = confusion_matrix(test_NN, predictions_NN)
    return predictions_NN, test_NN, cm

#print the result for Oversampling
print("a. Random Undersampling:")
predictions_RO, test_RO, cm_RO = undersampling("random")
print("Confusion Matrix: ")
print(cm_RO)
print("Accuracy:", accuracy_score(test_RO, predictions_RO))

print("\nb. Cluster Undersampling:")
predictions_Cluster, test_Cluster, cm_Cluster = undersampling("Cluster")
print("Confusion Matrix: ")
print(cm_Cluster)
print("Accuracy:", accuracy_score(test_Cluster, predictions_Cluster))

print("\nc. Tomek Undersampling:")
predictions_Tomek, test_Tomek, cm_Tomek = undersampling("Tomek")
print("Confusion Matrix: ")
print(cm_Tomek)
print("Accuracy:", accuracy_score(test_Tomek, predictions_Tomek),"\n")
