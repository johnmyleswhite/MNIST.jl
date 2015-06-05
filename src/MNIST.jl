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

    function imageheader(filename::String)
        open(filename, "r") do io
            magic_number = bswap(read(io, Uint32))
            total_items = bswap(read(io, Uint32))
            nrows = bswap(read(io, Uint32))
            ncols = bswap(read(io, Uint32))
            return magic_number, int(total_items), int(nrows), int(ncols)
	end
    end

    function labelheader(filename::String)
        open(filename, "r") do io
            magic_number = bswap(read(io, Uint32))
            total_items = bswap(read(io, Uint32))
            return magic_number, int(total_items)
        end
    end

    function getimage(filename::String, index::Integer)
        open(filename, "r") do io
            seek(io, IMAGEOFFSET + NROWS * NCOLS * (index - 1))
            return read(io, Uint8, (MNIST.NROWS, MNIST.NCOLS))'
        end
    end

    function getlabel(filename::String, index::Integer)
        open(filename, "r") do io
            seek(io, LABELOFFSET + (index - 1))
            label = read(io, Uint8)
            return label
        end
    end

    @compat trainimage(index::Integer) = map(Float64,getimage(TRAINIMAGES, index))
    @compat testimage(index::Integer) = map(Float64,getimage(TESTIMAGES, index))
    @compat trainlabel(index::Integer) = map(Float64,getlabel(TRAINLABELS, index))
    @compat testlabel(index::Integer) = map(Float64,getlabel(TESTLABELS, index))
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
