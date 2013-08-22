using MNIST

function preprocess(data)
    trainX, trainLabels = data
    trainX /= max(trainX)
    D, N = size(trainX)
    trainY = zeros(10, N)
    # Use 1-of-c encoding
    for n in 1:N
        trainY[trainLabels[n]+1, n] = 1
    end
    return trainX, trainY
end

function predict(W, b, X)
    D, N = size(X)
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
    E = sum(deltas.^2) / N
    return E, Wd, bd
end

trainX, trainY = preprocess(traindata())
W = randn(10, 784) * 0.05
b = randn(10, 1) * 0.05
alpha = 0.001
Y = predict(W, b, trainX)
for i in 1:100
    E, Wd, bd = gradient(W, b, trainX, trainY)
    W -= Wd * alpha
    b -= bd * alpha
    println(E)
end
testX, testY = preprocess(testdata())
prediction = predict(W, b, testX)
F, N = size(testY)
correct = 0
for n in 1:N
    v, i1 = findmax(prediction[:, n])
    v, i2 = findmax(testY[:, n])
    if i1 == i2
        correct += 1
    end
end
println(W)
println(b)
println(correct)

