import layers
import numpy as np

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

if __name__ == '__main__':
    np.set_printoptions(suppress=True)
    print("Part 3:")
    testingGradients()
    print("Part 4:")
    testObjectiveLayers()
