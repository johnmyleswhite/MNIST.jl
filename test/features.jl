
x = trainfeatures(1)
y = trainlabel(1)

@test isequal(size(x), (28 * 28, ))
@test isa(y, Float64)

x = testfeatures(1)
y = testlabel(1)

@test isequal(size(x), (28 * 28, ))
@test isa(y, Float64)

X, Y = traindata()

@test isequal(size(X), (28 * 28, 60_000))
@test isequal(size(Y), (60_000, ))

X, Y = testdata()

@test isequal(size(X), (28 * 28, 10_000))
@test isequal(size(Y), (10_000, ))

