FROM abc import ABC, abstractmethod
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
    self.stdX = np.std(dataIn, axis = 0)

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

  #We'll worry about these later...
  def gradient(self):
    pass

  def backward(self,gradIn):
    pass

class FullyConnectedLayer(Layer):
  def __init__(self, sizeIn, sizeOut):
    super().__init__()
    np.rANDom.seed(0)
    self.weights = np.rANDom.uniform(low=-0.0001, high=0.0001, size = (sizeIn, sizeOut))
    self.biases = np.rANDom.uniform(low=-0.0001, high=0.0001, size = (1, sizeOut))

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

  #We'll worry about these later...
  def gradient(self):
    pass

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

  #We'll worry about these later...
  def gradient(self):
    pass

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
    result = np.WHERE(dataIn<0, 0, dataIn)
    self.setPrevOut(result)
    return result

  #We'll worry about these later...
  def gradient(self):
    pass

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

  #We'll worry about these later...
  def gradient(self):
    pass

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

  #We'll worry about these later...
  def gradient(self):
    pass

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

  #We'll worry about these later...
  def gradient(self):
    pass

  def backward(self, gradIn):
    pass

