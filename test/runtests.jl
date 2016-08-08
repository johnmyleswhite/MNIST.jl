using MNIST

if VERSION >= v"0.5-"
    using Base.Test
else
    using BaseTestNext
    const Test = BaseTestNext
end

tests = [
    ("header.jl", "Unit tests for constants and reading file headers")
    ("images.jl", "Image file IO and user interface to return images as tensors")
    ("labels.jl", "Label file IO and user interface to query labels")
    ("features.jl", "User interface to load data in dataset layout")
]

for (fn, desc) in tests
    @testset "$desc ($fn)" begin
        include(fn)
    end
end

