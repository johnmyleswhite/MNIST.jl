using MNIST, Plots

trainX, trainY = traindata()

testX, testY = testdata()

mutable struct Mind
    layers::Vector{Int}
    weights::Vector{Matrix{Float32}}
    biases::Vector{Vector{Float32}}
    λ::Float32
end

function Mind(layers)
    ws = Vector{Matrix{Float32}}(undef, length(layers)-1)
    bs = Vector{Vector{Float32}}(undef, length(layers)-1)
    for (i, (nin, nout)) in enumerate(zip(layers[1:end-1], layers[2:end]))
        ws[i] = randn(nout, nin) / √nin
        bs[i] = randn(nout)
    end
    return Mind(layers, ws, bs, 0.01)
end

relu(X) = max.(X, 0)

function d(f::Function)
    if f==relu
        return max.(sign.(Z), 0)
    end
end

function softmax(X)
    E = exp.(X)
    M = sum(E, dims=1)
    return E ./ M
end

function backprop!(mind::Mind, X::Matrix{Float32}, Y::Matrix{Float32}, l=1)
    if l == length(mind.layers) - 1
        Z = softmax(mind.weights[l]*X .+ mind.biases[l])
        δ = (Z .- Y) / size(Z,2)
    else
        Z = relu(mind.weights[l]*X .+ mind.biases[l])
        δ = backprop!(mind, Z, Y, l+1)
    end
    mind.biases[l] .-= mind.λ * sum(δ, dims=2)
    mind.weights[l] .-=  mind.λ * (δ * X')
    dZ = max.(sign.(Z), 0)
    return mind.weights[l]' * (δ .* dZ)
end


function predict(mind::Mind, X::Matrix{Float32}, l=1)
    if l == length(mind.layers)-1
        return softmax(mind.weights[l]*X .+ mind.biases[l])
    else
        Z = relu(mind.weights[l]*X .+ mind.biases[l])
        return predict(mind, Z, l+1)
    end
end

score(P::Matrix{Float32}, Y::Matrix{Float32}) = -sum(Y .* log.(P .+ eps())) / size(Y,2)


function learn!(mind::Mind, X::Matrix{Float32}, Y::Matrix{Float32},  X2::Matrix{Float32}, Y2::Matrix{Float32})
    training_skorz = []
    test_skorz = []
    for cycle ∈ 1:512
        for goop in 1:300
            x = X[:, 100*(goop-1)+1:100*goop]
            y = Y[:, 100*(goop-1)+1:100*goop]
            backprop!(mind, x, y, 1)
        end
        push!(training_skorz, score(predict(mind, X), Y))
        push!(test_skorz, score(predict(mind, X2), Y2))
    end
    plot(hcat(training_skorz, test_skorz))
end

m = Mind([784, 256, 256, 10])

learn!(m, trainX, trainY, testX, testY)

sum(predict(m, testX) .* testY) / size(testX,2)