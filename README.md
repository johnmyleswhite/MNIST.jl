MNIST.jl
========

[![Build Status](https://travis-ci.org/johnmyleswhite/MNIST.jl.svg?branch=master)](https://travis-ci.org/johnmyleswhite/MNIST.jl)

# Introduction

This package provides access to the classic MNIST data set of
handwritten digits that has been used as a testbed for new
machine learning methods. The MNIST data set is included with
the package for convenience without any claim of copyright to
the images, which are the property of Yann LeCun and Corinna
Cortes. The images were downloaded into their original IDX
format from http://yann.lecun.com/exdb/mnist/ and are stored
in the `data/` directory.

To work with the data, you will typically want to store the digits
in Julian matrices. To load the i-th image or label, use:

* `trainimage(i)`
* `testimage(i)`
* `trainfeatures(i)`
* `testfeatures(i)`
* `trainlabel(i)`
* `testlabel(i)`

An images will be stored in 28x28 `UInt8` array, the features in
784-entry `Float64` vector and the label will be returned as a
`Float64` scalar. It is also possible to load multiple
observations at once,

```julia
julia> summary(trainimage(1:5))
"28×28×5 Array{UInt8,3}"

julia> summary(trainfeatures(1:5))
"784×5 Array{Float64,2}"

julia> summary(trainlabel(1:5))
"5-element Array{Float64,1}"
```

To access the entire data set at once, use:

* `traindata()`
* `testdata()`

The `traindata` method will return a tuple of two items: the first
element of the tuple will be a 784x60,000 `Float64` matrix
containing all of the images in the training set. The second element
of the tuple will be a 60,000 entry `Float64` vector containing
the labels of all of the images in the train set. The `testdata`
method will return an equivalent tuple for the test set, which contains
10,000 images instead of 60,000.

# Example Usage

	using MNIST
	features = trainfeatures(1)
	label = trainlabel(1)

	trainX, trainY = traindata()
	testX, testY = testdata()
