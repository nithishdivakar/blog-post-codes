import numpy as np
from sklearn.metrics import log_loss
import matplotlib.pyplot as plt

# 3D square's corners

X_train = np.asarray([
    [0,0,0],
    [0,0,1],
    [0,1,0],
    [0,1,1],
    [1,0,0],
    [1,0,1],
    [1,1,0],
    [1,1,1],
])

X_train[X_train==0] = -1

Y_train = np.asarray([
    [1,0],
    [0,1],
    [1,0],
    [0,1],
    [1,0],
    [0,1],
    [1,0],
    [0,1],
])

from sklearn.utils import shuffle
def get_next_batch(X,Y,batch_size):
    sample_size = X.shape[0]
    
    while True:
      X,Y = shuffle(X,Y,random_state=0)
      for idx in range(0,sample_size,batch_size):
        yield X[idx:idx+batch_size], Y[idx:idx+batch_size]
      
        



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


def softmax_forward(z):
  M = np.max(z,axis=-1,keepdims=True)
  e = np.exp(z-M) # normalisation trick so 
                  # that largest of z is 0
  sigma = e.sum(axis=-1,keepdims=True)
  s = e/sigma
  return s


def matmul(x,W):
  return np.matmul(x,W.T)

def bias_add(xW,b):
  return xW+b

feature_size = X_train.shape[1]
class_size   = Y_train.shape[1]
B = 5
get_batch = get_next_batch(X_train, Y_train, batch_size=B)
lr=0.01
LOSS = []

def init_params():
    THETA = {}
    def init_fn(shape):
        p = 1
        for s in shape:
            p*=s
        var = 1/p
        return np.random.normal(0,var,shape)
        #return np.random.normal(0,0.1,shape)
        
    THETA['W1'] = init_fn((5,3))#np.random.normal(0,0.1,(5,3))
    THETA['b1'] = init_fn((5,)) #np.random.normal(0,0.1,(5,))
    THETA['W2'] = init_fn((4,5))#np.random.normal(0,0.1,(4,5))
    THETA['b2'] = init_fn((4,)) #np.random.normal(0,0.1,(4,))
    THETA['W3'] = init_fn((2,4))#np.random.normal(0,0.1,(2,4))
    THETA['b3'] = init_fn((2,)) #np.random.normal(0,0.1,(2,))
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

def backward_pass(P,Q):
    
    GRAD = {}
    B = P['a0'].shape[0]
    
       
    dz3_da2 = Q['W3']
    da2_dz2 = relu_derv(P['z2'])
    dz2_dW2 = P['a1']
    
    dz2_da1 = Q['W2']
    da1_dz1 = relu_derv(P['z1'])
    dz1_dW1 = P['a0']
    
     
    dl_dz3  = (P['s']-P['y'])
    dl_da2  = np.dot(dl_dz3,dz3_da2)
    dl_dz2  = np.multiply(dl_da2,da2_dz2)
    
    dl_da1  = np.dot(dl_dz2,dz2_da1)
    dl_dz1  = np.multiply(dl_da1,da1_dz1)
    
    
    dl_dW3 = (1/B)*np.dot(dl_dz3.T,P['a2'])
    dl_db3 = (1/B)*dl_dz3.sum(axis=0)
    
    dl_dW2 = (1/B)*np.dot(dl_dz2.T,P['a1'])
    dl_db2 = (1/B)*dl_dz2.sum(axis=0)
    
    dl_dW1 = (1/B)*np.dot(dl_dz1.T,P['a0'])
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

def relu_derv(z):
    t = z.copy()
    t[t>0] = 1.0
    t[t<=0]= 0.0
    return t

Q = init_params()
for i in range(10000):
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
        
plt.figure()
plt.plot(LOSS)
plt.savefig("loss.png")
