module MNIST
    using Compat

    export trainfeatures, testfeatures,
           trainlabel, testlabel,
           traindata, testdata

    const IMAGEOFFSET = 16
    const LABELOFFSET = 8
    const NROWS = 28
    const NCOLS = 28
    const TRAINIMAGES = Pkg.dir("MNIST", "data", "train-images.idx3-ubyte")
    const TRAINLABELS = Pkg.dir("MNIST", "data", "train-labels.idx1-ubyte")
    const TESTIMAGES = Pkg.dir("MNIST", "data", "t10k-images.idx3-ubyte")
    const TESTLABELS = Pkg.dir("MNIST", "data", "t10k-labels.idx1-ubyte")

    function traindata()
        return features, labels
    end

    function testdata()
        return features, labels
    end
    function imageheader(filename::String)
        io = open(filename, "r")
        magic_number = bswap(read(io, Uint32))
        total_items = bswap(read(io, Uint32))
        nrows = bswap(read(io, Uint32))
        ncols = bswap(read(io, Uint32))
        close(io)
        return magic_number, Int(total_items), Int(nrows), Int(ncols)
    end

    function labelheader(filename::String)
        io = open(filename, "r")
        magic_number = bswap(read(io, Uint32))
        total_items = bswap(read(io, Uint32))
        close(io)
        return magic_number, Int(total_items)
    end

    function getimage(filename::String, index::Integer)
        io = open(filename, "r")
        seek(io, IMAGEOFFSET + NROWS * NCOLS * (index - 1))
        image = zeros(Uint8, NROWS, NCOLS)
        for i in 1:NROWS
            for j in 1:NCOLS
                image[i, j] = read(io, Uint8)
            end
        end
        close(io)
        return image
    end

    function getlabel(filename::String, index::Integer)
        io = open(filename, "r")
        seek(io, LABELOFFSET + (index - 1))
        label = read(io, Uint8)
        close(io)
        return label
    end

    trainimage(index::Integer) = map(Float64,getimage(TRAINIMAGES, index))
    testimage(index::Integer) = map(Float64,getimage(TESTIMAGES, index))
    trainlabel(index::Integer) = map(Float64,getlabel(TRAINLABELS, index))
    testlabel(index::Integer) = map(Float64,getlabel(TESTLABELS, index))
    trainfeatures(index::Integer) = vec(trainimage(index))
    testfeatures(index::Integer) = vec(testimage(index))

    function traindata()
        _, nimages, nrows, ncols = imageheader(TRAINIMAGES)
        features = Array(Float64, nrows * ncols, nimages)
        labels = Array(Float64, nimages)
        for index in 1:nimages
            features[:, index] = trainfeatures(index)
            labels[index] = trainlabel(index)
        end
        return features, labels
    end

    function testdata()
        _, nimages, nrows, ncols = imageheader(TESTIMAGES)
        features = Array(Float64, nrows * ncols, nimages)
        labels = Array(Float64, nimages)
        for index in 1:nimages
            features[:, index] = testfeatures(index)
            labels[index] = testlabel(index)
        end
        return features, labels
    end
end # module
