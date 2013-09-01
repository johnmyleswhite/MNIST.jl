using NumericExtensions

# Multinomial logistic regression

type MLR
    W::Array{Float64,2}
    b::Array{Float64,2}

    function MLR(D::Int64, F::Int64)
        new(zeros(F, D), zeros(F, 1))
    end
end

function update(model::MLR, momentums::Array{Any,1})
    model.W += momentums[1]
    model.b += momentums[2]
end

function getDims(model::MLR)
    {size(model.W), size(model.b)}
end

function predict(model::MLR,
                 X::Array{Float64,2})
    A = model.W * X
    broadcast(+, A, model.b)
    softmax(A, 1)
end

function gradient(model::MLR,
                  X::Array{Float64,2},
                  T::Array{Float64,2})
    N = size(T, 2)
    deltas = predict(model, X) - T
    Wd = deltas * X' / N
    bd = sum(deltas, 2) / N
    MSE = sum(deltas.^2) / (2 * N)
    MSE, {Wd, bd}
end

function sgd(model::MLR,
             X::Array{Float64,2},
             Y::Array{Float64,2},
             numEpochs::Integer,
             alpha::Real = 0.1,
             eta::Real = 0.5,
             batchSize::Integer = 32)
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

function evaluate(model::MLR,
                  X::Array{Float64,2},
                  Y::Array{Float64,2})
    prediction = predict(model, X)
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
