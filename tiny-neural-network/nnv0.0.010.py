import numpy as np
from sklearn.metrics import log_loss
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
np.random.seed(123456)

lr = 0.01
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

def get_next_batch(X,Y,batch_size):
  sample_size = X.shape[0]
  
  while True:
    X,Y = shuffle(X,Y,random_state=0)
    for idx in range(0,sample_size,batch_size):
      yield X[idx:idx+batch_size], Y[idx:idx+batch_size]
      
        



def init_fn(shape):
    p = 1
    for s in shape:
        p*=s
    var = 1/p
    return np.random.normal(0,var,shape)

class Layer(object):
    def __init__(self):
        self.params = {}
        
    def forward(self,inputs):
        pass
    
    def backward(self,future_gradients, inputs):
        pass
    
    def update_params(self, new_params):
        # sanity checks
        for key in new_params:
#             print(key)
            assert key in self.params
            assert new_params[key].shape==self.params[key].shape
        
        # actual update
        for key in new_params:
            self.params[key] = new_params[key]
    
class DenseLayer(Layer):
    def __init__(self, name, input_shape, output_shape):
        super().__init__()
        self.name = name
        self.params['W'] = init_fn(shape=(output_shape,input_shape))
        self.params['b'] = init_fn(shape=(output_shape,))
        
    def forward(self,x):
        def matmul(x,W):
            return np.dot(x,W.T)
        def bias_add(xW,b):
            return xW+b

        z = bias_add(matmul(x,self.params['W']), self.params['b'])
        return z
    
    def backward(self,dl_dz, a):
        GRAD = {}
        B = a.shape[0]
        GRAD["{}:grad_W".format(self.name)] = (1/B)*np.dot(dl_dz.T,a)
        GRAD["{}:grad_b".format(self.name)] = (1/B)*dl_dz.sum(axis=0)
        GRAD["{}:grad_a".format(self.name)] = np.dot(dl_dz,self.params['W'])
        return GRAD
  
class ReluLayer(Layer):
    def __init__(self, name):
        self.name = name
        
    def forward(self,z):
        a = z.copy()
        a[a<0]=0.0
        return a
    
    def backward(self, dl_da, z):
        GRAD = {}
        da_dz = z.copy()
        da_dz[da_dz>0] = 1.0
        da_dz[da_dz<=0]= 0.0
        GRAD["{}:grad_z".format(self.name)] = np.multiply(dl_da,da_dz)
        return GRAD
    
class SoftmaxCrossEntropyLayer(Layer):
    def __init__(self, name):
        self.name = name
        
    def forward(self,z,y):
        def softmax(z):
            M = np.max(z,axis=-1,keepdims=True)
            e = np.exp(z-M) # normalisation trick so 
                          # that largest of z is 0
            sigma = e.sum(axis=-1,keepdims=True)
            s = e/sigma
            return s
        
        s    = softmax(z) 
        loss = log_loss(y_true=y, y_pred=s)
        return loss,s
        
    def backward(self,s,y):
        GRAD = {}
        GRAD["{}:grad_z".format(self.name)] = (s-y)
        return GRAD






DenseL1 = DenseLayer("DL1",3,5)
ReluL1  = ReluLayer("RL1")
DenseL2 = DenseLayer("DL2",5,4)
ReluL2  = ReluLayer("RL2")
DenseL3 = DenseLayer("DL3",4,2)
SCL     = SoftmaxCrossEntropyLayer("SCL")
        


    

feature_size = X_train.shape[1]
class_size   = Y_train.shape[1]
B = 5
get_batch = get_next_batch(X_train, Y_train, batch_size=B)
LOSS = []

for i in range(10000):
    x_b,y_b = next(get_batch)

    a0 = x_b
    z1 = DenseL1.forward(a0)
    a1 = ReluL1.forward(z1)
    z2 = DenseL2.forward(a1)
    a2 = ReluL2.forward(z2)
    z3 = DenseL3.forward(a2)
    l,s = SCL.forward(z3,y_b)
    LOSS.append(l)


    GRAD_SCL = SCL.backward(s,y_b)    
    GRAD_DenseL3 = DenseL3.backward(GRAD_SCL['SCL:grad_z'],a2) 
    GRAD_ReluL2  = ReluL2.backward(GRAD_DenseL3["DL3:grad_a"],z2)
    GRAD_DenseL2 = DenseL2.backward(GRAD_ReluL2["RL2:grad_z"],a1)
    GRAD_ReluL1  = ReluL1.backward(GRAD_DenseL2["DL2:grad_a"],z1)
    GRAD_DenseL1 = DenseL1.backward(GRAD_ReluL1["RL1:grad_z"],a0)

    DenseL3.update_params({'W':DenseL3.params['W'] - lr*GRAD_DenseL3["DL3:grad_W"],'b':DenseL3.params['b'] - lr*GRAD_DenseL3["DL3:grad_b"]})
    DenseL2.update_params({'W':DenseL2.params['W'] - lr*GRAD_DenseL2["DL2:grad_W"],'b':DenseL2.params['b'] - lr*GRAD_DenseL2["DL2:grad_b"]})
    DenseL1.update_params({'W':DenseL1.params['W'] - lr*GRAD_DenseL1["DL1:grad_W"],'b':DenseL1.params['b'] - lr*GRAD_DenseL1["DL1:grad_b"]})
    #break

plt.figure()
plt.plot(LOSS)
plt.savefig("loss.png")
