import numpy as np
np.random.seed(12345)
import matplotlib.pyplot as plt
from keras.datasets import cifar10,mnist
from sklearn.metrics import log_loss
from sklearn.utils import shuffle


## data ops
def get_next_batch(X,Y,batch_size):
  sample_size = X.shape[0]
  while True:
    X,Y = shuffle(X,Y,random_state=0)
    for idx in range(0,sample_size,batch_size):
      yield X[idx:idx+batch_size], Y[idx:idx+batch_size]

def class_to_one_hot(arr):
  n = arr.shape[0]
  k = arr.max()+1
  one_hot = np.zeros((n,k))
  one_hot[np.arange(n),arr] = 1.0
  return one_hot

## param ops
def param_initializer(shape):
  return np.random.normal(0,0.01,shape)


## forward ops
def relu(z):
  a = z.copy()
  a[a<0]=0.0
  return a

def identity(z):
  return z

def matmul(x,W):
  return np.matmul(x,W.T)

def bias_add(xW,b):
  return xW+b

def softmax(z):
  M = np.max(z,axis=-1,keepdims=True)
  e = np.exp(z-M) # normalisation trick so 
                  # that largest of z is 0
  sigma = e.sum(axis=-1,keepdims=True)
  s = e/sigma
  return s

## backward ops

def relu_derv(z):
  t = z.copy()
  t[t>0] = 1.0
  t[t<=0]= 0.0
  return t


def init_params():
  THETA = {}
  THETA['W1'] = param_initializer((500,28*28))
  THETA['b1'] = param_initializer((500,)) 
  THETA['W2'] = param_initializer((500,500))
  THETA['b2'] = param_initializer((500,))
  THETA['W3'] = param_initializer((10,500))
  THETA['b3'] = param_initializer((10,))
  return THETA

def forward_pass(x_b,y_b,Q):
  P = {}
  P['a0'] = x_b
  P['z1'] = bias_add(matmul(P['a0'],Q['W1']), Q['b1'])
  P['a1'] = relu(P['z1'])

  P['z2'] = bias_add(matmul(P['a1'],Q['W2']), Q['b2'])
  P['a2'] = relu(P['z2'])
  
  P['z3'] = bias_add(matmul(P['a2'],Q['W3']), Q['b3'])
  P['s']  = softmax(P['z3'])
  
  P['y']  = y_b
  P['L']  = log_loss(y_true=P['y'], y_pred=P['s'])
  return P

def backward_pass(ACTIVATIONS,PARAMS):
  
  GRAD = {}
  B = ACTIVATIONS['a0'].shape[0]
  
     
  dz3_da2 = PARAMS['W3']
  da2_dz2 = relu_derv(ACTIVATIONS['z2'])
  dz2_dW2 = ACTIVATIONS['a1']
  
  dz2_da1 = PARAMS['W2']
  da1_dz1 = relu_derv(ACTIVATIONS['z1'])
  dz1_dW1 = ACTIVATIONS['a0']
  
   
  dl_dz3  = (ACTIVATIONS['s']-ACTIVATIONS['y'])
  dl_da2  = np.dot(dl_dz3,dz3_da2)
  dl_dz2  = np.multiply(dl_da2,da2_dz2)
  
  dl_da1  = np.dot(dl_dz2,dz2_da1)
  dl_dz1  = np.multiply(dl_da1,da1_dz1)
  
  
  dl_dW3 = (1/B)*np.dot(dl_dz3.T,ACTIVATIONS['a2'])
  dl_db3 = (1/B)*dl_dz3.sum(axis=0)
  
  dl_dW2 = (1/B)*np.dot(dl_dz2.T,ACTIVATIONS['a1'])
  dl_db2 = (1/B)*dl_dz2.sum(axis=0)
  
  dl_dW1 = (1/B)*np.dot(dl_dz1.T,ACTIVATIONS['a0'])
  dl_db1 = (1/B)*dl_dz1.sum(axis=0)
  

  GRAD['dl_dW3'] = dl_dW3
  GRAD['dl_db3'] = dl_db3
  GRAD['dl_dW2'] = dl_dW2
  GRAD['dl_db2'] = dl_db2
  GRAD['dl_dW1'] = dl_dW1
  GRAD['dl_db1'] = dl_db1
  
  
  ## extra
  
  GRAD['dz3_da2']=dz3_da2
  GRAD['da2_dz2']=da2_dz2
  GRAD['dz2_dW2']=dz2_dW2
  GRAD['dz2_da1']=dz2_da1
  GRAD['da1_dz1']=da1_dz1
  GRAD['dz1_dW1']=dz1_dW1
  GRAD['dl_dz3']=dl_dz3
  
  return GRAD


B = 32
lr= 0.01

(x_train, y_train), (x_test, y_test) = mnist.load_data()

X_train = x_train.reshape(x_train.shape[0],-1)/255
X_test  = x_test.reshape(x_test.shape[0],-1)/255
Y_train = class_to_one_hot(y_train.reshape(y_train.shape[0]))
Y_test  = class_to_one_hot(y_test.reshape(y_test.shape[0]))


get_batch = get_next_batch(X_train, Y_train, batch_size=B)
LOSS = []

Q = init_params()
for idx in range(5000):
  x_b,y_b = next(get_batch)
  P = forward_pass(x_b,y_b,Q)
  LOSS.append(P['L'])
  GRAD = backward_pass(P,Q)
  
  Q['W3'] = Q['W3'] - lr*GRAD['dl_dW3']
  Q['b3'] = Q['b3'] - lr*GRAD['dl_db3']
  Q['W2'] = Q['W2'] - lr*GRAD['dl_dW2']
  Q['b2'] = Q['b2'] - lr*GRAD['dl_db2']
  Q['W1'] = Q['W1'] - lr*GRAD['dl_dW1']
  Q['b1'] = Q['b1'] - lr*GRAD['dl_db1']
  if idx%100==0:
      print("#",end ='')
plt.figure()
plt.plot(LOSS)
plt.savefig("loss.png")
