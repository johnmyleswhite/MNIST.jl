include("utils.jl")
using NumericExtensions
using MNIST

# Multinomial logistic regression

type MLR <: Model
    W::Array{Float64,2}
    b::Array{Float64,2}

    function MLR(D::Integer, F::Integer)
        new(zeros(F, D), zeros(F, 1))
    end
end

function update(model::MLR, momentums::Vector)
    model.W += momentums[1]
    model.b += momentums[2]
end

function getDims(model::MLR)
    {size(model.W), size(model.b)}
end

function predict(model::MLR,
                 X::Matrix)
    A = model.W * X
    broadcast(+, A, model.b)
    Z = softmax(A, 1)
    Z, {Z}
end

function gradient(model::MLR,
                  X::Matrix,
                  T::Matrix)
    N = size(T, 2)
    Y, outputs = predict(model, X)
    deltas = Y - T
    Wd = deltas * X' / N
    bd = sum(deltas, 2) / N
    MSE = sum(deltas.^2) / (2 * N)
    MSE, {Wd, bd}
end

function demo_mlr(numEpochs::Integer = 10,
                  alpha::Real = 0.1,
                  eta::Real = 0.5,
                  batchSize::Integer = 32)
    trainX, trainY = preprocess(traindata())
    D = size(trainX, 1)
    F = size(trainY, 1)
    model = MLR(D, F)
    model = sgd(model, trainX, trainY, numEpochs, alpha, eta, batchSize)

    testX, testY = preprocess(testdata())
    correct, accuracy = score(model, testX, testY)
    accuracy *= 100
    println("$correct correct predictions ($accuracy% accuracy) on test set")
end
