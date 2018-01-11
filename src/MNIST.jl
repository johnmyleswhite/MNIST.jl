module MNIST
    export trainfeatures, testfeatures,
           trainlabel, testlabel,
           traindata, testdata

    const IMAGEOFFSET = 16
    const LABELOFFSET = 8

    const NROWS = 28
    const NCOLS = 28

    const TRAINIMAGES = joinpath(
        dirname(@__FILE__), "..", "data", "train-images.idx3-ubyte"
    )
    const TRAINLABELS = joinpath(
        dirname(@__FILE__), "..", "data", "train-labels.idx1-ubyte"
    )
    const TESTIMAGES = joinpath(
        dirname(@__FILE__), "..", "data", "t10k-images.idx3-ubyte"
    )
    const TESTLABELS = joinpath(
        dirname(@__FILE__), "..", "data", "t10k-labels.idx1-ubyte"
    )

    function imageheader(filename::AbstractString)
        open(filename, "r") do io
            magic_number = bswap(read(io, UInt32))
            total_items = bswap(read(io, UInt32))
            nrows = bswap(read(io, UInt32))
            ncols = bswap(read(io, UInt32))
            return (
                magic_number,
                Int(total_items),
                Int(nrows),
                Int(ncols)
            )
        end
    end

    function labelheader(filename::AbstractString)
        open(filename, "r") do io
            magic_number = bswap(read(io, UInt32))
            total_items = bswap(read(io, UInt32))
            return magic_number, Int(total_items)
        end
    end

    function getimage(filename::AbstractString, index::Integer)
        open(filename, "r") do io
            seek(io, IMAGEOFFSET + NROWS * NCOLS * (index - 1))
            image_t = read(io, UInt8, (MNIST.NROWS, MNIST.NCOLS))
            return image_t'
        end
    end

    function getlabel(filename::AbstractString, index::Integer)
        open(filename, "r") do io
            seek(io, LABELOFFSET + (index - 1))
            label = read(io, UInt8)
            return label
        end
    end

    function trainimage(index::Integer)
        convert(Matrix{Float64}, getimage(TRAINIMAGES, index))
    end

    function testimage(index::Integer)
        convert(Matrix{Float64}, getimage(TESTIMAGES, index))
    end

    function trainlabel(index::Integer)
        convert(Float64, getlabel(TRAINLABELS, index))
    end

    function testlabel(index::Integer)
        convert(Float64, getlabel(TESTLABELS, index))
    end

    trainfeatures(index::Integer) = vec(trainimage(index))

    testfeatures(index::Integer) = vec(testimage(index))

    function traindata()
        _, nimages, nrows, ncols = imageheader(TRAINIMAGES)
        features = Matrix{Float64}(nrows * ncols, nimages)
        labels = Vector{Float64}(nimages)
        for index in 1:nimages
            features[:, index] = trainfeatures(index)
            labels[index] = trainlabel(index)
        end
        return features, labels
    end

    function testdata()
        _, nimages, nrows, ncols = imageheader(TESTIMAGES)
        features = Matrix{Float64}(nrows * ncols, nimages)
        labels = Vector{Float64}(nimages)
        for index in 1:nimages
            features[:, index] = testfeatures(index)
            labels[index] = testlabel(index)
        end
        return features, labels
    end
end # module
