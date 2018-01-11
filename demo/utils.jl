abstract type Model end

function preprocess{R}(data::Tuple{Matrix{R}, Vector{R}})
    trainX, trainLabels = data
    trainX /= max(trainX)
    N = size(trainX, 2)
    trainY = zeros(Float64, 10, N)
    # Use 1-of-c encoding
    for n in 1:N
        trainY[trainLabels[n]+1, n] = 1
    end
    trainX, trainY
end

function sgd!(model::Model,
             X::Matrix{Float64},
             Y::Matrix{Float64},
             numEpochs::Integer,
             alpha::Real = 0.1,
             eta::Real = 0.5,
             batchSize::Integer = 32)
    N = size(X, 2)
    momentums = [zeros(Float64, d) for d in size(model)]
    for i in 1:numEpochs
        MSE = 0.0
        indices = shuffle(1:N)
        for n in 1:batchSize:N
            sub_indices = view(indices, n:min(n+batchSize-1, N))
            tmpMSE, grads = gradient(model, view(X, :, sub_indices), view(Y, :, sub_indices))
            MSE += tmpMSE
            momentums = [eta * m - alpha * g for (m, g) in zip(momentums, grads)]
            update!(model, momentums)
        end
        MSE /= fld((N + batchSize - 1), batchSize)
        println("Epoch $i, MSE = $MSE")
    end
    model
end

function score(model::Model,
               X::Matrix{Float64},
               Y::Matrix{Float64})
    prediction, outputs = predict(model, X)
    N = size(Y, 2)
    correct = 0
    for n in 1:N
        c1 = indmax(view(prediction, :, n))
        c2 = indmax(view(Y, :, n))
        if c1 == c2
            correct += 1
        end
    end
    accuracy = correct / N
    correct, accuracy
end
