import layers
import numpy as np
import math
import matplotlib.pyplot as plt

def testFullyConnectedGradient(X):
    fullyConnectedLayer = layers.FullyConnectedLayer(3, 2)
    fullyConnectedLayer.setWeights(np.array([[1, 2], [3, 4], [5, 6]]))
    fullyConnectedLayer.setBiases(np.array([-1, 2]))
    print("Fully connected Layer: ")
    fullyConnectedLayer.forward(X)
    print(fullyConnectedLayer.gradient())
    print("\n")

def testLinearLayerGradient(X):
    linearLayer = layers.LinearLayer()
    print("Linear Layer: ")
    linearLayer.forward(X)
    print(linearLayer.gradient())
    print("\n")

def testReLuLayerGradient(X):
    reluLayer = layers.ReLuLayer()
    print("ReLu Layer: ")
    reluLayer.forward(X)
    print(reluLayer.gradient())
    print("\n")

def testLogisticSigmoidLayerGradient(X):
    sigmoidLayer = layers.LogisticSigmoidLayer()
    print("Logistic Sigmoid Layer: ")
    sigmoidLayer.forward(X)
    print(sigmoidLayer.gradient())
    print("\n")

def testSoftmaxLayerGradient(X):
    softmaxLayer = layers.SoftmaxLayer()
    print("Softmax Layer: ")
    softmaxLayer.forward(X)
    print(softmaxLayer.gradient())
    print("\n")

def testTanhLayerGradient(X):
    tanhLayer = layers.TanhLayer()
    print("TanH Layer: ")
    tanhLayer.forward(X)
    print(tanhLayer.gradient())
    print("\n")

def readCSV(fileName):
    dataIn = np.genfromtxt(fileName, delimiter=',')
    return dataIn[1:]

def testingGradients():
    X = np.array([[1, 2, 3], [4, 5, 6]])
    testFullyConnectedGradient(X) 
    testLinearLayerGradient(X)
    testReLuLayerGradient(X)
    testLogisticSigmoidLayerGradient(X)
    testSoftmaxLayerGradient(X)
    testTanhLayerGradient(X)

def testObjectiveLayers():
    sq = layers.SquaredError()
    print(" == Squared Error == ")
    print("Eval: ", sq.eval(np.array([[0],[1]]), np.array([[0.2],[0.3]])))
    print("Gradient: ")
    print(sq.gradient(np.array([[0],[1]]), np.array([[0.2],[0.3]])))
    print()

    ls = layers.LogLoss()
    print(" == Log Loss == ")
    print("Eval:", ls.eval(np.array([[0],[1]]), np.array([[0.2],[0.3]])))
    print("Gradient:")
    print(ls.gradient(np.array([[0],[1]]), np.array([[0.2],[0.3]])))
    print()

    print(" == Cross Entropy == ")
    crossEntropy = layers.CrossEntropy()
    print("Eval:", crossEntropy.eval(np.array([[1, 0, 0], [0, 1, 0]]), np.array([[0.2, 0.2, 0.6], [0.2, 0.7, 0.1]])))
    print("Gradient: ")
    print(crossEntropy.gradient(np.array([[1, 0, 0], [0, 1, 0]]), np.array([[0.2, 0.2, 0.6], [0.2, 0.7, 0.1]])))

def linearRegression():
    np.finfo(dtype=np.float32)
    X = readCSV("medical.csv")
    Y = X[:, [-1]].copy()
    X = np.delete(X, -1, axis=1)   

    L1 = layers.InputLayer(X)
    L2 = layers.FullyConnectedLayer(X.shape[1], Y.shape[1])
    L3 = layers.SquaredError()
    layerOrder = [L1, L2, L3]
    #forwards!
    h = X
    Yhat = h
    eval = []
    epochItr = []
    for epoch in range(10000):
        h = X
        for i in range(len(layerOrder)-1):
            h = layerOrder[i].forward(h)
        
        print(epoch)
        eval.append(layerOrder[-1].eval(Y,h))
        if len(eval) > 1 and abs(eval[-1] - eval[-2]) < math.pow(10, -10):
            print("accuracy: " + str(accuracy(Y,h)))
            break
        #backwards!
        grad = layerOrder[-1].gradient(Y,h)
        Yhat = h
        epochItr.append(epoch)
        for i in range(len(layerOrder)-2,0,-1):
            newgrad= layerOrder[i].backward(grad)
            if(isinstance(layerOrder[i],layers.FullyConnectedLayer)):
                layerOrder[i].updateWeights(grad,math.pow(10,-4))
            grad = newgrad
    
    print("accuracy: " + str(accuracy(Y,Yhat)))
    print("RMSE: " + str(RMSE(Y, Yhat)))
    print("SMAPE: " + str(SMAPE(Y, Yhat)))
    plt.plot(epochItr, eval)
    plt.xlabel('epoch')
    plt.ylabel('MSE')
    plt.show()

def logLossRun():
    np.finfo(dtype=np.float32)
    X = readCSV("KidCreative.csv")
    Y = X[:, [1]].copy()
    X = np.delete(X, 1, axis=1)   

    L1 = layers.InputLayer(X)
    L2 = layers.FullyConnectedLayer(X.shape[1], Y.shape[1])
    L3 = layers.LogisticSigmoidLayer()
    L4 = layers.LogLoss()
    layerOrder = [L1, L2, L3, L4]
    
    h = X
    threshold = math.pow(10, -10)
    loglossArray = []
    epochArray = []
    Yhat = h
    for epoch in range (10000):
        h = X
        print("epoch:",epoch)
        for i in range(len(layerOrder) - 1):
            h = layerOrder[i].forward(h)
        
        loss = layerOrder[-1].eval(Y, h)
        
        grad = layerOrder[-1].gradient(Y, h)
        loglossArray = np.append(loglossArray, loss)
        epochArray = np.append(epochArray, epoch)
        
        change = loglossArray[-2] - loglossArray[-1] if epoch > 2 else 0
        
        if (0 < change < threshold):
            print(accuracy(Y, h))
            break
        
        for i in range(len(layerOrder)-2,0,-1):
            newgrad = layerOrder[i].backward(grad)
            
            if(isinstance(layerOrder[i],layers.FullyConnectedLayer)):
                layerOrder[i].updateWeights(grad,math.pow(10,-4))
            
            grad = newgrad
        Yhat = h

    print("Accuracy: " + str(accuracy(Y,np.round(Yhat))))
    ax = plt.axes()
    plt.xlabel("epoch")
    plt.ylabel("Log Loss")
    ax.plot(epochArray, loglossArray)
    plt.show()

def accuracy(Y, h):
    return np.mean(Y == h)

def RMSE(Y, h):
    return np.sqrt(np.mean(np.square(Y - h)))


def SMAPE(Y, h):
    return np.mean(np.abs(Y - h) / (np.abs(Y) + np.abs(h)))

if __name__ == '__main__':
    # linearRegression()
    logLossRun()
