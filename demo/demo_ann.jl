include("ann.jl")
include("utils.jl")
using MNIST

numEpochs = 10 # Number of training epochs
alpha = 0.1    # Learning rate

trainX, trainY = preprocess(traindata())
trainX = trainX[:, 1:10000]
trainY = trainY[:, 1:10000]
D = size(trainX, 1)
F = size(trainY, 1)

ann = ANN([D, 200, F], 0.05)
for i in 1:10
    println("Episode $i")
    gradients = gradient(ann, trainX, trainY)
    L = size(gradients, 1)
    for l in 1:L
        setWeights(ann, l, getWeights(ann, l) - alpha * gradients[l])
    end
end

prediction, outputs = predict(ann, trainX)
N = size(trainY, 2)
correct = 0
for n in 1:N
    c1 = indmax(prediction[:, n])
    c2 = indmax(trainY[:, n])
    if c1 == c2
        correct += 1
    end
end
accuracy = correct / N
println("$correct, $accuracy")
