using MNIST

function preprocess(data)
    trainX, trainLabels = data
    trainX /= max(trainX)
    N = size(trainX, 2)
    trainY = zeros(10, N)
    # Use 1-of-c encoding
    for n in 1:N
        trainY[trainLabels[n]+1, n] = 1
    end
    return trainX, trainY
end

function predict(W, b, X)
    N = size(X, 2)
    # Predict everything at once
    Y = W * X
    broadcast(+, Y, b)
    # We use a trick here to avoid NaNs: we substract the maximum of each row
    # (this is mathematically equivalent)
    for n in 1:N
        Y[:, n] = exp(Y[:, n] - max(Y[:, n]))
        Y[:, n] /= sum(Y[:, n])
    end
    return Y
end

function gradient(W, b, X, T)
    F, N = size(T)
    Y = predict(W, b, X)
    deltas = Y-T
    Wd = deltas * X' / N
    bd = zeros(size(b))
    for n in 1:N
        bd += deltas[:, n]
    end
    bd /= N
    MSE = sum(deltas.^2) / (2 * N)
    return MSE, Wd, bd
end


alpha = 0.1    # Learning rate
eta = 0.5      # Momentum

trainX, trainY = preprocess(traindata())
D, N = size(trainX)
F = size(trainY, 1)
numEpochs = 20
batchSize = 32
stdDev = 0.05
W = randn(F, D) * stdDev
b = randn(F, 1) * stdDev
momentumW = zeros(F, D)
momentumb = zeros(F, 1)

for i in 1:numEpochs
    MSE = 0.0
    indices = shuffle([1:N])
    for n in 1:batchSize:N
        tmpMSE, Wd, bd = gradient(W, b,
            trainX[:, indices[n:min(n+batchSize-1, end)]],
            trainY[:, indices[n:min(n+batchSize-1, end)]])
        MSE += tmpMSE
        momentumW = eta * momentumW - alpha * Wd
        momentumb = eta * momentumb - alpha * bd
        W += momentumW
        b += momentumb
    end
    MSE /= fld((N + batchSize - 1), batchSize)
    println("Epoch $i, MSE = $MSE")
end

testX, testY = preprocess(testdata())
prediction = predict(W, b, testX)
N = size(testY, 2)
correct = 0
for n in 1:N
    v, i1 = findmax(prediction[:, n])
    v, i2 = findmax(testY[:, n])
    if i1 == i2
        correct += 1
    end
end
println("$correct/$N correct predictions on test set")
