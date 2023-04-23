import layers
import numpy as np
import sys

def testFullyConnectedLayers(X):
    fullyConnectedLayer = layers.FullyConnectedLayer(4, 2)
    fullyConnectedLayer.setWeights(np.array([[1, 2], [2, 0], [1, 1], [-1, 4]]))
    fullyConnectedLayer.setBiases(np.array([1, 2]))
    print("Fully connected Layer: ")
    print(fullyConnectedLayer.forward(X))
    print("\n")

def testLinearLayer(X):
    linearLayer = layers.LinearLayer()
    print("Linear Layer: ")
    print(linearLayer.forward(X))
    print("\n")

def testReLuLayer(X):
    reluLayer = layers.ReLuLayer()
    print("ReLu Layer: ")
    print(reluLayer.forward(X))
    print("\n")

def testLogisticSigmoidLayer(X):
    sigmoidLayer = layers.LogisticSigmoidLayer()
    print("Logistic Sigmoid Layer: ")
    print(sigmoidLayer.forward(X))
    print("\n")

def testSoftmaxLayer(X):
    softmaxLayer = layers.SoftmaxLayer()
    print("Softmax Layer: ")
    print(softmaxLayer.forward(X))
    print("\n")

def testTanhLayer(X):
    tanhLayer = layers.TanhLayer()
    print("TanH Layer: ")
    print(tanhLayer.forward(X))
    print("\n")

def testInputLayer(X):
    inputLayer = layers.InputLayer(X)
    print("Input Layer: ")
    print(inputLayer.forward(X))
    print("\n")

def readCSV():
    dataIn = np.genFROMtxt('mcpd_augmented.csv', delimiter=',')
    return dataIn[1:]

def testingFullDataSet():
    X = readCSV()
    inputData = layers.InputLayer(X)
    fullConectedData = layers.FullyConnectedLayer(X.shape[1], 2)
    sigmoidData = layers.LogisticSigmoidLayer()

    connectLayers = [inputData, fullConectedData, sigmoidData]

    h = X    
    for i in range(len(connectLayers)):
        h = connectLayers[i].forward(h)

    print(h)

def testingLayers():
    X = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
    testInputLayer(X)
    testFullyConnectedLayers(X) 
    testLinearLayer(X)
    testReLuLayer(X)
    testLogisticSigmoidLayer(X)
    testSoftmaxLayer(X)
    testTanhLayer(X)

def testConnectingLayers():
    X = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
    inputData = layers.InputLayer(X)
    fullConectedData = layers.FullyConnectedLayer(4, 2)
    fullConectedData.setWeights(np.array([[1, 2], [2, 0], [1, 1], [-1, 4]]))
    fullConectedData.setBiases(np.array([1, 2]))
    sigmoidData = layers.LogisticSigmoidLayer()

    connectLayers = [inputData, fullConectedData, sigmoidData]
    printLayer = ["Input-> ", "Full Connected->", "Logistic Sigmoid->"]

    h = X    
    for i in range(len(connectLayers)):
        h = connectLayers[i].forward(h)
        print(printLayer[i])
        print(h)
        print("")

    print("Yhat")
    print(h)
    print("")

if __name__ == '__main__':
    print("Part 3:")
    testingLayers()
    print("Part 4:")
    testConnectingLayers()
    print("Part 5:")
    testingFullDataSet()
