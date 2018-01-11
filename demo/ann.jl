include("utils.jl")
using NumericExtensions
using MNIST

"""
Artificial neural network
"""
struct ANN <: Model
    w::Vector{Float64}
    dims::Vector{Int}
    L::Int

    function ANN(dims::Vector, stdDev::Real)
        numWeights = 0
        L = length(dims) - 1
        for l in 1:L
            numWeights += (dims[l]+1) * dims[l+1]
        end
        w = randn(numWeights) * stdDev
        new(w, dims, L)
    end
end

function update!(ann::ANN, momentums::Vector)
    for l in 1:ann.L
        setweights!(ann, l, weights(ann, l) + momentums[l])
    end
    ann
end

function Base.size(ann::ANN)
    [(ann.dims[l+1], ann.dims[l]+1) for l in 1:ann.L]
end

function weights(ann::ANN, layer::Integer)
    offset = 1
    for l in 1:layer-1
        offset += (ann.dims[l]+1) * ann.dims[l+1]
    end
    reshape(view(ann.w, offset:offset+(ann.dims[layer]+1)*ann.dims[layer+1]-1),
            ann.dims[layer+1], ann.dims[layer]+1)
end

function setweights!(ann::ANN, layer::Integer, W::Matrix)
    offset = 1
    for l in 1:layer-1
        offset += (ann.dims[l]+1) * ann.dims[l+1]
    end
    ann.w[offset:offset+(ann.dims[layer]+1)*ann.dims[layer+1]-1] = W[:]
    ann
end

function predict(ann::ANN,
                 X::Matrix)
    outputs = Vector{Matrix{Float64}}()
    Z = X
    for l in 1:ann.L
        Z = vcat(Z, ones(Float64, 1, size(Z, 2)))
        push!(outputs, Z)
        W = weights(ann, l)
        A = W * Z
        if l < ann.L
            Z = tanh(A)
        else
            Z = softmax(A, 1)
        end
    end
    push!(outputs, Z)
    Z, outputs
end

function gradient(ann::ANN,
                  X::Matrix,
                  T::Matrix)
    gradients = Vector{Matrix{Float64}}()
    N = size(X, 2)
    Y, outputs = predict(ann, X)
    dEdX = Y - T
    MSE = sum(dEdX.^2) / (2 * N)
    for l in ann.L:-1:1
        W = weights(ann, l)
        Z = outputs[l+1]
        X = outputs[l]
        if l == ann.L
            Deltas = dEdX
        else
            Zd = 1 - Z .* Z
            Deltas = Zd .* dEdX
            Deltas = Deltas[1:end-1, :] # Get rid of bias
        end
        unshift!(gradients, Deltas * X' / N)
        if l > 1
          dEdX = W' * Deltas
        end
    end
    MSE, gradients
end

function demo_ann(numEpochs::Integer = 10,
                  alpha::Real = 0.1,
                  eta::Real = 0.5,
                  batchSize::Integer = 32,
                  stdDev::Real = 0.05,
                  H::Vector = [100])
    trainX, trainY = preprocess(traindata())
    D = size(trainX, 1)
    F = size(trainY, 1)
    model = ANN([D, H, F], stdDev)
    model = sgd!(model, trainX, trainY, numEpochs, alpha, eta, batchSize)

    testX, testY = preprocess(testdata())
    correct, accuracy = score(model, testX, testY)
    println("$correct correct predictions ($(accuracy*100)% accuracy) on test set")
end
