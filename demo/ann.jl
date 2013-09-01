using NumericExtensions

type ANN
    w::Array{Float64,1}
    dims::Array{Int64,1}
    L::Int64

    function ANN(dims::Array{Int64,1}, stdDev::Float64)
        numWeights = 0
        L = size(dims, 1) - 1
        for l in 1:L
            numWeights += (dims[l]+1) * dims[l+1]
        end
        w = randn(numWeights) * stdDev
        new(w, dims, L)
    end
end

function numLayers(ann::ANN)
    size(ann.dims, 1) - 1
end

function getWeights(ann::ANN, layer::Integer)
    offset = 1
    for l in 1:layer-1
        offset += (ann.dims[l]+1) * ann.dims[l+1]
    end
    reshape(unsafe_view(ann.w, offset:offset+(ann.dims[layer]+1)*ann.dims[layer+1]-1),
            ann.dims[layer+1], ann.dims[layer]+1)
end

function setWeights(ann::ANN, layer::Integer, W::Array{Float64,2})
    offset = 1
    for l in 1:layer-1
        offset += (ann.dims[l]+1) * ann.dims[l+1]
    end
    ann.w[offset:offset+(ann.dims[layer]+1)*ann.dims[layer+1]-1] = W[:]
end

function predict(ann::ANN,
                 X::Array{Float64,2})
    outputs = {}
    Z = X
    for l in 1:ann.L
        Z = vcat(Z, ones(1, size(Z, 2)))
        push!(outputs, Z)
        W = getWeights(ann, l)
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
                  X::Array{Float64,2},
                  T::Array{Float64,2})
    gradients = {}
    N = size(X, 2)
    Y, outputs = predict(ann, X)
    dEdX = Y - T
    for l in ann.L:-1:1
        W = getWeights(ann, l)
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
    gradients
end
