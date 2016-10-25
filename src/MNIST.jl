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

    function imageheader(io::IO)
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

    function labelheader(io::IO)
        magic_number = bswap(read(io, UInt32))
        total_items = bswap(read(io, UInt32))
        return magic_number, Int(total_items)
    end

    function getrawimage(io::IO, index::Integer)
        seek(io, IMAGEOFFSET + NROWS * NCOLS * (index - 1))
        image_t = read(io, UInt8, (MNIST.NROWS, MNIST.NCOLS))
        return image_t'
    end

    function getrawlabel(io::IO, index::Integer)
        seek(io, LABELOFFSET + (index - 1))
        label = read(io, UInt8)
        return label
    end

    function getimage(io::IO, index::Integer)
        convert(Array{Float64}, getrawimage(io, index))
    end

    function getlabel(io::IO, index::Integer)
      convert(Float64, getrawlabel(io, index))
    end

    getfeatures(io::IO, index::Integer) = vec(getimage(io, index))

    trainfeatures(index::Integer) =
        open(io -> vec(getimage(io, index)), TRAINIMAGES)

    testfeatures(index::Integer) =
        open(io -> vec(getimage(io, index)), TESTIMAGES)

    trainlabel(index::Integer) =
        open(io -> getlabel(io, index), TRAINLABELS)

    testlabel(index::Integer) =
        open(io -> getlabel(io, index), TESTLABELS)

    function traindata()
        io = IOBuffer(read(TRAINIMAGES))
        labelio = IOBuffer(read(TRAINLABELS))
        _, nimages, nrows, ncols = imageheader(io)
        features = Array(Float64, nrows * ncols, nimages)
        labels = Array(Float64, nimages)
        for index in 1:nimages
            features[:, index] = getfeatures(io, index)
            labels[index] = getlabel(labelio, index)
        end
        return features, labels
    end

    function testdata()
        io = IOBuffer(read(TESTIMAGES))
        labelio = IOBuffer(read(TESTLABELS))
        _, nimages, nrows, ncols = imageheader(io)
        features = Array(Float64, nrows * ncols, nimages)
        labels = Array(Float64, nimages)
        for index in 1:nimages
            features[:, index] = getfeatures(io, index)
            labels[index] = getlabel(labelio, index)
        end
        return features, labels
    end
end # module
