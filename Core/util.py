import numpy as np
import pandas as pd
import scipy
import matplotlib.pyplot as plt
from copy import deepcopy
from sklearn import preprocessing
import timeit
from enum import Enum,auto
from scipy.special import expit

def crossValidate(data:np.ndarray, split1, split2):
    data = np.asarray(data)
    m = data.__len__()
    idx = np.random.permutation(m)
    data = data[idx]
    return data[:int(split1*m),:], data[int(split1*m):int((split1+split2)*m),:], data[int((split1+split2)*m):,:], idx

def kfold(data:np.ndarray, k):
    m = data.__len__()
    idx = np.random.permutation(m)
    kfoldData = []
    for i in range(k):
        training = [x for j, x in enumerate(idx) if j % k != i]
        validation = [x for j, x in enumerate(idx) if j % k == i]
        train = data[training]
        test = data[validation]
        kfoldData.append([train,test])
    return np.asarray(kfoldData), idx

def update_line(hl, ax1, new_Xdata, new_Ydata):
    if new_Xdata == 0: pass
    hl.set_xdata(np.append(hl.get_xdata(), new_Xdata))
    hl.set_ydata(np.append(hl.get_ydata(), new_Ydata))
    ax1 = plt.gca()
    ax1.relim()
    ax1.autoscale_view()
    plt.draw()
    plt.pause(0.0001)

def activation(w,x):
    return np.dot(w,x)

def sigmoid(w,x):
  return 1 / (1 + np.exp(-(np.dot(w,x))))

def predict(w, x):
    return 1 if activation(w,x)>=0 else 0

def confusionMatrix(X,Y,W):
    confusion = np.zeros((2,2))
    for i,x in enumerate(X):
        prediction = predict(x,W)
        if (prediction == 1 and Y[i] == 1): confusion[0, 0] += 1
        elif (prediction == 0 and Y[i] == 1): confusion[0, 1] += 1
        elif (prediction == 1 and Y[i] == 0): confusion[1, 0] += 1
        elif (prediction == 0 and Y[i] == 0): confusion[1, 1] += 1
    return confusion

def scale(x):
    return preprocessing.scale(x)

def multiClassPredict(X,Y,W):
    accuracy = 0

    for x,y in zip(X,Y):
        y1 = activation(W[0],x)
        y2 = activation(W[1],x)
        y3 = activation(W[2],x)
        prediction = 1

        if (y1 >= y2 and y1 >= y3): prediction = 1
        elif (y2 >= y1 and y2 >= y3): prediction = 2
        elif (y3 >= y1 and y3 >= y2): prediction = 3

        if (y[0] == prediction): accuracy+=1

    return float(accuracy/len(X))*100

def SVD_compress(x,accuracy):
    m,n = x.shape
    [U, S, V] = scipy.linalg.svd(x, full_matrices=False)
    # print("Result of close test by numpy package for new data vectors and original one: ",np.allclose(x, np.dot(U * S, V)))
    sigmaSumThreshold = float(accuracy * np.sum(S))
    sum = 0
    i = 0
    while (sum < sigmaSumThreshold):
        sum += S[i]
        i += 1
    featurelength = i
    S = S[0:i]
    U = U[:, 0:i]
    V = V[0:i, :]
    print("Dimension reduction with SVD went from {} to {} when {}% of eigen values have already saved.".format(n,featurelength,accuracy * 100))
    return  U * S, [U,S,V]

def normalize(x):
    x_scaled = deepcopy(x)
    for i in range(x.shape[1]):
        feature = x_scaled[:, i]
        max = np.max(feature)
        min = np.min(feature)
        try:
            x_scaled[:, i] = (feature - min) / (max - min)
        except ZeroDivisionError:
            pass
    return x_scaled