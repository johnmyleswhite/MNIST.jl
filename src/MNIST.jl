module MNIST

export

    trainimage,
    testimage,

    trainlabel,
    testlabel,

    trainfeatures,
    testfeatures,

    traindata,
    testdata

# Constants

const IMAGEOFFSET = 16
const LABELOFFSET = 8

const DATA_PATH = joinpath(dirname(@__FILE__), "..", "data")
const TRAINIMAGES = joinpath(DATA_PATH, "train-images.idx3-ubyte")
const TRAINLABELS = joinpath(DATA_PATH, "train-labels.idx1-ubyte")
const TESTIMAGES  = joinpath(DATA_PATH, "t10k-images.idx3-ubyte")
const TESTLABELS  = joinpath(DATA_PATH, "t10k-labels.idx1-ubyte")

# Includes

include("header.jl")
include("images.jl")
include("labels.jl")
include("features.jl")

end

