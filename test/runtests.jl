using MNIST
using Base.Test

tests = [
    ("header.jl", "Unit tests for constants and reading file headers")
    ("features.jl", "User interface to load data in dataset layout")
]

for (fn, desc) in tests
    @testset "$desc ($fn)" begin
        include(fn)
    end
end
