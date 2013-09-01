include("mlr.jl")
include("utils.jl")
using MNIST

numEpochs = 10 # Number of training epochs
alpha = 0.1    # Learning rate
eta = 0.5      # Momentum
batchSize = 32 # Size of mini-batches

trainX, trainY = preprocess(traindata())
D, N = size(trainX)
F = size(trainY, 1)
model = MLR(D, F)
model = sgd(model, trainX, trainY, numEpochs, alpha, eta, batchSize)

testX, testY = preprocess(testdata())
correct, accuracy = evaluate(model, testX, testY)
accuracy *= 100
println("$correct correct predictions ($accuracy% accuracy) on test set")
