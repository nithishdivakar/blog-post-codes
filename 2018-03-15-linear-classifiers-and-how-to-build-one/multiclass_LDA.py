import keras
from scipy import linalg
from keras.datasets import mnist, cifar10, cifar100, imdb, fashion_mnist
import numpy as np

def get_linear_classifier(X,Y,cls):
  mu = {}
  p  = {}
  w  = {}
  b  = {}
  # classwise mean and probabilties
  for c in cls:
    mu[c] = np.mean(X[Y==c],axis=0)
    p[c] = (Y==c).sum()/X.shape[0]

  # common covariance matrix and its inverse
  M = np.empty_like(X)
  for c in cls:
    M[Y==c,:] = mu[c]

  S  = np.cov((X-M).T)/X.shape[0] 
  S_inv = linalg.pinv(S)
  
  # classifier parameters
  for c in cls:
    w[c] = S_inv.dot(mu[c])
    b[c] = np.log(p[c]) - 0.5* mu[c].T.dot(S_inv).dot(mu[c]) 
  return w,b

def test_model(w,b,X,Y):
  W = np.zeros((X.shape[1],len(w)))
  B = np.zeros((len(b),))
  for c in w:
    W[:,c] = w[c]
    B[c]   = b[c]
  pred = np.argmax(X.dot(W)+B,axis=1) 
  #print(pred,Y)
  acc  = sum(pred==Y)/Y.shape[0]
  return acc

#"""
## MNIST
print("[INFO] MNIST Dataset" )

(x_train, y_train), (x_test, y_test) = mnist.load_data()

X_train = x_train.reshape(x_train.shape[0],-1)
X_test  = x_test.reshape(x_test.shape[0],-1)

w,b = get_linear_classifier(X_train,y_train,[0,1,2,3,4,5,6,7,8,9]) 

acc  = test_model(w,b,X_train,y_train)
print("[INFO] train accuracy = {:0.4f}".format(acc))

acc  = test_model(w,b,X_test,y_test)
print("[INFO] test accuracy = {:0.4f}".format(acc))
#"""

#"""
## Fashion-MNIST
print("[INFO] Fashion MNIST Dataset" )

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

X_train = x_train.reshape(x_train.shape[0],-1)
X_test  = x_test.reshape(x_test.shape[0],-1)

w,b = get_linear_classifier(X_train,y_train,[0,1,2,3,4,5,6,7,8,9]) 

acc  = test_model(w,b,X_train,y_train)
print("[INFO] train accuracy = {:0.4f}".format(acc))

acc  = test_model(w,b,X_test,y_test)
print("[INFO] test accuracy = {:0.4f}".format(acc))
#"""

#"""
## Cifar10
print("[INFO] CIFAR 10 Dataset" )

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

X_train = x_train.reshape(x_train.shape[0],-1)
X_test  = x_test.reshape(x_test.shape[0],-1)
y_train = y_train.reshape(y_train.shape[0])
y_test = y_test.reshape(y_test.shape[0])

w,b = get_linear_classifier(X_train,y_train,[0,1,2,3,4,5,6,7,8,9]) 

acc  = test_model(w,b,X_train,y_train)
print("[INFO] train accuracy = {:0.4f}".format(acc))

acc  = test_model(w,b,X_test,y_test)
print("[INFO] test accuracy = {:0.4f}".format(acc))

#"""

#"""
## Cifar100
print("[INFO] CIFAR 10 Dataset" )

(x_train, y_train), (x_test, y_test) = cifar100.load_data(label_mode='fine')

X_train = x_train.reshape(x_train.shape[0],-1)
X_test  = x_test.reshape(x_test.shape[0],-1)
y_train = y_train.reshape(y_train.shape[0])
y_test = y_test.reshape(y_test.shape[0])

w,b = get_linear_classifier(X_train,y_train,range(100)) 

acc  = test_model(w,b,X_train,y_train)
print("[INFO] train accuracy = {:0.4f}".format(acc))

acc  = test_model(w,b,X_test,y_test)
print("[INFO] test accuracy = {:0.4f}".format(acc))
#"""

