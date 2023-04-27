from abc import ABC, abstractmethod
import numpy as np

class Layer(ABC):
  def __init__(self):
    self.__prevIn = []
    self.__prevOut = []

  def setPrevIn(self, dataIn):
    self.__prevIn = dataIn
  
  def setPrevOut(self, out):
    self.__prevOut = out
  
  def getPrevIn(self):
    return self.__prevIn
  
  def getPrevOut(self):
    return self.__prevOut

  @abstractmethod
  def forward(self, dataIn):
    pass

  @abstractmethod
  def gradient(self):
    pass

  @abstractmethod
  def backward(self, gradIn):
    pass  

class InputLayer(Layer):
  #Input:  dataIn, an NxD matrix
  #Output:  None
  def __init__(self, dataIn):
    self.meanX = np.mean(dataIn, axis = 0) 
    self.stdX = np.std(dataIn, axis = 0, ddof=1)

  #Input:  dataIn, an NxD matrix
  #Output: An NxD matrix
  def forward(self, dataIn):
    self.setPrevIn(dataIn)
    zScoredData = [[0] * len(dataIn[0]) for _ in range(len(dataIn))]
    for i in range(len(dataIn)):
      for j in range(len(dataIn[0])):
        zScoredData[i][j] = (dataIn[i][j] - self.meanX[j]) / self.stdX[j]
    self.setPrevOut(zScoredData)
    return zScoredData

  def gradient(self):
    pass

  def backward(self,gradIn):
    pass

class FullyConnectedLayer(Layer):
  def __init__(self, sizeIn, sizeOut):
    super().__init__()
    np.random.seed(0)
    self.weights = np.random.uniform(low=-0.0001, high=0.0001, size = (sizeIn, sizeOut))
    self.biases = np.random.uniform(low=-0.0001, high=0.0001, size = (1, sizeOut))

  #Input:  None
  #Output: The sizeIn x sizeOut weight matrix.
  def getWeights(self):
    return self.weights

  #Input: The sizeIn x sizeOut weight matrix.
  #Output: None
  def setWeights(self, weights):
    self.weights = np.array(weights)

  #Input:  The 1 x sizeOut bias vector
  #Output: None
  def getBiases(self):
    return self.biases

  #Input:  None
  #Output: The 1 x sizeOut biase vector
  def setBiases(self, biases):
    self.biases = np.array(biases)

  #Input:  dataIn, an NxD data matrix
  #Output:  An NxK data matrix
  def forward(self, dataIn):
    self.setPrevIn(dataIn)
    result = np.dot(dataIn, self.weights) + self.biases
    self.setPrevOut(result)
    return result

  def gradient(self):
    X = self.getWeights()
    result = []
    for _ in range(X.shape[1]):
      result.append(X.transpose())
    return result

  def backward(self, gradIn):
    pass

class LinearLayer(Layer):
  #Input:  None
  #Output:  None
  def __init__(self):
    super().__init__()

  #Input:  dataIn, an NxK matrix
  #Output:  An NxK matrix
  def forward(self, dataIn):
    self.setPrevIn(dataIn)
    self.setPrevOut(dataIn)
    return dataIn

  def gradient(self):
    x = self.getPrevOut()
    result = []
    for _ in range(x.shape[0]):
      result.append(np.identity(self.getPrevOut().shape[1], dtype=int))
    return result

    
  def backward(self,gradIn):
    pass

class ReLuLayer(Layer):
  #Input:  None
  #Output:  None
  def __init__(self):
    super().__init__()

  #Input:  dataIn, an NxK matrix
  #Output:  An NxK matrix
  def forward(self, dataIn):
    self.setPrevIn(dataIn)
    result = np.where(dataIn<0, 0, dataIn)
    self.setPrevOut(result)
    return result

  def gradient(self):
    x = self.getPrevOut()
    result = []
    for _ in range(x.shape[0]):
      result.append(np.identity(self.getPrevOut().shape[1], dtype=int))
    return result

  def backward(self, gradIn):
    pass

class LogisticSigmoidLayer(Layer):
  #Input:  None
  #Output:  None
  def __init__(self):
    super().__init__()

  #Input:  dataIn, an NxK matrix
  #Output:  An NxK matrix
  def forward(self, dataIn):
    self.setPrevIn(dataIn)
    result = 1/ (1 + np.exp(-dataIn))
    self.setPrevOut(result)
    return result

  def gradient(self):
    x = self.getPrevOut()
    result = []
    for i in range(x.shape[0]):
      y = []
      for j in range(x.shape[1]):
        y.append(x[i][j]*(1-x[i][j]))
      result.append(np.diag(y))
    return result

  def backward(self, gradIn):
    pass

class SoftmaxLayer(Layer):
  #Input:  None
  #Output:  None
  def __init__(self):
    super().__init__()

  #Input:  dataIn, an NxK matrix
  #Output:  An NxK matrix
  def forward(self, dataIn):
    self.setPrevIn(dataIn)
    e_x = np.exp(dataIn - np.max(dataIn))
    result = e_x / e_x.sum(axis=1, keepdims=True)
    self.setPrevOut(result)
    return result

  def gradient(self):
    X = self.getPrevOut()
    N = X.shape[0]
    K = X.shape[1]
    result = np.zeros((N, K, K))
        
    for k in range(N):
      for i in range(K):
        for j in range(K):
          if(i == j):
            result[k, i, j] = X[k, i] * (1 - X[k, i])
          else:
            result[k, i, j] = -X[k, i] * X[k, j]
    return result

  def backward(self, gradIn):
    pass

class TanhLayer(Layer):
  #Input:  None
  #Output:  None
  def __init__(self):
    super().__init__()

  #Input:  dataIn, an NxK matrix
  #Output:  An NxK matrix
  def forward(self, dataIn):
    self.setPrevIn(dataIn)
    result = (np.exp(dataIn) - np.exp(-dataIn)) / (np.exp(dataIn) + np.exp(-dataIn))
    self.setPrevOut(result)
    return result

  def gradient(self):
    x = self.getPrevOut()
    result = []
    for i in range(x.shape[0]):
      y = []
      for j in range(x.shape[1]):
        y.append(1-(x[i][j]**2))
      result.append(np.diag(y))
    return result

  def backward(self, gradIn):
    pass

eps = 10 ** -7

class SquaredError():
  def eval(self, Y, Yhat):
    return np.square(Y - Yhat).mean()
  
  def gradient(self, Y, Yhat):
    return -2*np.subtract(Y, Yhat)

class LogLoss():
  def eval(self, Y, Yhat):
    y = np.multiply(Y, np.log(Yhat + eps))
    ynot = np.multiply(1 - Y, np.log(1 - Yhat + eps))
    return -np.add(y, ynot).mean()
  
  def gradient(self, Y, Yhat):
    num = np.subtract(Y, Yhat)
    den = np.multiply(Yhat, 1-Yhat)

    return -np.divide(num, den + eps)

class CrossEntropy():
  def eval(self, Y, Yhat):
    return -np.mean(np.sum(np.multiply(Y, np.log(Yhat)+eps), axis = 1))
  
  def gradient(self, Y, Yhat):
    return -np.divide(Y, Yhat + eps)