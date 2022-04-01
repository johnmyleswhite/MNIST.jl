using MNIST, Colors


trainX, trainY = traindata()

Gray.(reshape(trainX[:, 6578], 28, 28))


mutable struct Mind
    layers::Vector{Int}
    weights::Vector{Matrix{Float32}}
    biases::Vector{Vector{Float32}}
end

function Mind(layers)
    ws = Vector{Matrix{Float32}}(undef, length(layers)-1)
    bs = Vector{Vector{Float32}}(undef, length(layers)-1)
    for (i, (nin, nout)) in enumerate(zip(layers[1:end-1], layers[2:end]))
        ws[i] = randn(nout, nin)
        bs[i] = randn(nout)
    end
    return Mind(layers, ws, bs )
end

function train(mind::Mind, X::Matrix{Float32}, Y::Matrix{Float32}, l=1)
    if l == length(mind.layers)
        return X
    else
        Z = abs.(mind.weights[l]*X .+ mind.biases[l])
        return train(mind, Z, Y, l+1)
    end
end

m = Mind([784, 100, 100, 10])

train(m, trainX, trainY)