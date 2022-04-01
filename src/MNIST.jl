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
        io = open(filename, "r")
        magic_number = bswap(read(io, UInt32))
        total_items = bswap(read(io, UInt32))
        nrows = bswap(read(io, UInt32))
        ncols = bswap(read(io, UInt32))
        close(io)
        return (
            magic_number,
            Int(total_items),
            Int(nrows),
            Int(ncols)
        )
    end

    function labelheader(filename::AbstractString)
        io = open(filename, "r")
        magic_number = bswap(read(io, UInt32))
        total_items = bswap(read(io, UInt32))
        close(io)
        return magic_number, Int(total_items)
    end

    function getimage(filename::AbstractString, index::Integer)
        io = open(filename, "r")
        seek(io, IMAGEOFFSET + NROWS * NCOLS * (index - 1))
        image_t = [read(io, UInt8) for col ∈ 1:MNIST.NCOLS, row ∈ 1:MNIST.NROWS]
        close(io)
        return image_t' ./ 255
    end

    function getlabel(filename::AbstractString, index::Integer)
        io = open(filename, "r")
        seek(io, LABELOFFSET + (index - 1))
        label = read(io, UInt8)
        close(io)
        label_maker = zeros(10)
        label_maker[label+1] = 1.0
        return label_maker
    end

    function trainimage(index::Integer)
        convert(Array{Float32}, getimage(TRAINIMAGES, index))
    end

    function testimage(index::Integer)
        convert(Array{Float32}, getimage(TESTIMAGES, index))
    end

    function trainlabel(index::Integer)
        convert(Array{Float32}, getlabel(TRAINLABELS, index))
    end

    function testlabel(index::Integer)
        convert(Array{Float32}, getlabel(TESTLABELS, index))
    end

    trainfeatures(index::Integer) = vec(trainimage(index))

    testfeatures(index::Integer) = vec(testimage(index))

    function traindata()
        _, nimages, nrows, ncols = imageheader(TRAINIMAGES)
        features = Matrix{Float32}(undef, nrows * ncols, nimages)
        labels = Matrix{Float32}(undef, 10, nimages)
        for index in 1:nimages
            features[:, index] .= trainfeatures(index)
            labels[:, index] .= trainlabel(index)
        end
        return features, labels
    end

    function testdata()
        _, nimages, nrows, ncols = imageheader(TESTIMAGES)
        features = Matrix{Float32}(undef, nrows * ncols, nimages)
        labels = Matrix{Float32}(undef, 10, nimages)
        for index in 1:nimages
            features[:, index] .= testfeatures(index)
            labels[:, index] .= testlabel(index)
        end
        return features, labels
    end
end # module

