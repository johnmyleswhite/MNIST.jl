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
    Wd = deltas * X'
    bd = zeros(size(b))
    for n in 1:N
        bd += deltas[:, n]
    end
    Wd /= N
    bd /= N
    E = sum(deltas.^2) / (2 * N)
    return E, Wd, bd
end

trainX, trainY = preprocess(traindata())
D, N = size(trainX)
F = size(trainY, 1)
W = randn(F, D) * 0.05
b = randn(F, 1) * 0.05
alpha = 0.1
numEpochs = 10

for i in 1:numEpochs
    E, Wd, bd = gradient(W, b, trainX, trainY)
    W -= Wd * alpha
    b -= bd * alpha
    println("Epoch $i, MSE = $E")
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

