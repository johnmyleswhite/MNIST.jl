abstract Model

function preprocess{R}(data::(Array{R,2},Array{R,1}))
    trainX, trainLabels = data
    trainX /= max(trainX)
    N = size(trainX, 2)
    trainY = zeros(10, N)
    # Use 1-of-c encoding
    for n in 1:N
        trainY[trainLabels[n]+1, n] = 1
    end
    trainX, trainY
end

function sgd(model::Model,
             X::Array{Float64,2},
             Y::Array{Float64,2},
             numEpochs::Integer,
             alpha::Real = 0.1,
             eta::Real = 0.5,
             batchSize::Integer = 32)
    N = size(X, 2)
    momentums = {zeros(d) for d in getDims(model)}
    for i in 1:numEpochs
        MSE = 0.0
        indices = shuffle([1:N])
        for n in 1:batchSize:N
            tmpMSE, grads = gradient(model,
                X[:, indices[n:min(n+batchSize-1, end)]],
                Y[:, indices[n:min(n+batchSize-1, end)]])
            MSE += tmpMSE
            momentums = {eta * m - alpha * g for (m, g) in zip(momentums, grads)}
            update(model, momentums)
        end
        MSE /= fld((N + batchSize - 1), batchSize)
        println("Epoch $i, MSE = $MSE")
    end
    model
end

function score(model::Model,
                  X::Array{Float64,2},
                  Y::Array{Float64,2})
    prediction, outputs = predict(model, X)
    N = size(Y, 2)
    correct = 0
    for n in 1:N
        c1 = indmax(prediction[:, n])
        c2 = indmax(Y[:, n])
        if c1 == c2
            correct += 1
        end
    end
    accuracy = correct / N
    correct, accuracy
end
