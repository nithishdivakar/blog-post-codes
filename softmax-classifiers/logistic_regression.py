import keras
from scipy import linalg
from keras.datasets import  cifar10
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle as dataset_shuffle
from sklearn.metrics import log_loss


def get_predictions(x,W):
  z = np.matmul(x,W.T)
  M = np.max(z,axis=-1,keepdims=True)
  e = np.exp(z-M) # normalisation trick so 
                  # that largest of z is 0
  sigma = e.sum(axis=-1,keepdims=True)
  s = e/sigma
  return s

def train_classifier(
  X_train, Y_train,
  X_valid, Y_valid, 
  batch_size = 32, 
  lr = 0.001, 
  total_iterations = 10000
):

  feature_size = X_train.shape[1]
  class_size   = Y_train.shape[1] 
  

  def initialise_weights():
    return np.random.normal(0,0.01,size=(class_size,feature_size))

  def get_next_batch(X,Y,batch_size):
    sample_size = X.shape[0]
    while True:
      for idx in range(0,sample_size,batch_size):
        yield X[idx:idx+batch_size], Y[idx:idx+batch_size]

  get_batch = get_next_batch(X_train, Y_train, batch_size)
  
  def get_batch_gradient(x_b,y_b,s_b):
    g_b = np.matmul((s_b-y_b).T,x_b)
    return g_b/batch_size

  def update_weights(W,g_b):
    return W - lr*g_b

  def loop_hook(no_iterations, W,l_b, log_freq=1000):
    if no_iterations %(log_freq) == 0:
      acc = test_model(W,X_valid,Y_valid) # Running Validation
      print("[INFO] # = {:7d}  Valid Acc. = {:4.4f} Loss = {:e}".format(
        no_iterations,
        acc,
        l_b
      ))
  
  loop_index = 1
  W = initialise_weights()
  while True:
    x_b,y_b = next(get_batch)
    s_b     = get_predictions(x_b,W)
    l_b     = log_loss(y_true=y_b, y_pred=s_b) 
    g_b     = get_batch_gradient(x_b,y_b,s_b)
    W       = update_weights(W,g_b)

    loop_hook(loop_index,W,l_b)
    if loop_index >= total_iterations: break
    loop_index = loop_index + 1

  return W

def test_model(W,X,Y):
  z = X.dot(W.T)
  pred = np.argmax(z,axis=1) 
  tru  = np.argmax(Y,axis=1)
  acc  = sum(pred==tru)/Y.shape[0]
  return acc

def class_to_one_hot(arr):
  n = arr.shape[0]
  k = arr.max()+1
  one_hot = np.zeros((n,k))
  one_hot[np.arange(n),arr] = 1.0
  return one_hot


#"""
# Cifar10
print("[INFO] CIFAR 10 Dataset" )

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

X_train = x_train.reshape(x_train.shape[0],-1)/255
X_test  = x_test.reshape(x_test.shape[0],-1)/255
Y_train = class_to_one_hot(y_train.reshape(y_train.shape[0]))
Y_test  = class_to_one_hot(y_test.reshape(y_test.shape[0]))

X_train,X_valid,Y_train,Y_valid = train_test_split(
                                    X_train,Y_train,
                                    test_size = 0.05,
                                    random_state=1634
                                  )

"""
# Abelation Study
B = [4,8,16,32,64,128,256,512]
LR =[1e-1,1e-2,1e-3,1e-4,1e-5]

for b,lr in product(B,LR):
  print("[INFO] lr = {:e}  batch size = {:3d}".format(lr,b))
  train_classifier(
    X_train, Y_train,
    X_valid, Y_valid, 
    batch_size = b, 
    lr=lr, 
    total_iterations = 10000
  )
#"""

# Acutal Training
W = train_classifier(
    X_train, Y_train,
    X_valid, Y_valid, 
    batch_size = 256, 
    lr=1e-4, 
    total_iterations = 1000000
  )

acc  = test_model(W,X_train,Y_train)
print("[INFO] train accuracy = {:0.4f}".format(acc))

acc  = test_model(W,X_test,Y_test)
print("[INFO] test accuracy = {:0.4f}".format(acc))
