using MNIST

x = trainfeatures(1)
y = trainlabel(1)

@assert isequal(size(x), (28 * 28, ))
@assert isa(y, Float64)

x = testfeatures(1)
y = testlabel(1)

@assert isequal(size(x), (28 * 28, ))
@assert isa(y, Float64)

X, Y = traindata()

@assert isequal(size(X), (28 * 28, 60_000))
@assert isequal(size(Y), (60_000, ))

X, Y = testdata()

@assert isequal(size(X), (28 * 28, 10_000))
@assert isequal(size(Y), (10_000, ))
