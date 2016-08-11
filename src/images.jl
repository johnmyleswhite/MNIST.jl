function readimage_raw!(buffer::Matrix{UInt8}, io::IO, index::Integer, nrows::Integer, ncols::Integer)
    seek(io, IMAGEOFFSET + nrows * ncols * (index - 1))
    read!(io, buffer)
end


"""
    readimage_raw(io::IO, index::Integer, nrows::Integer, ncols::Integer)

Jumps to the position of `io` where the bytes for the `index`'th
image are located and reads the next `nrows` * `ncols` bytes. The
read bytes are returned as a `Matrix{UInt8}` of size `(nrows, ncols)`.
"""
function readimage_raw(io::IO, index::Integer, nrows::Integer, ncols::Integer)
    buffer = Array(UInt8, nrows, ncols)
    readimage_raw!(buffer, io, index, nrows, ncols)
end


"""
    readimage_raw(io::IO, indices::AbstractVector, nrows::Integer, ncols::Integer)

Reads the first `nrows` * `ncols` bytes for each image index in
`indices` and stores them in a `Array{UInt8,3}` of size
`(nrows, ncols, length(indices))` in the same order as denoted
by `indices`.
"""
function readimage_raw(io::IO, indicies::AbstractVector, nrows::Integer, ncols::Integer)
    images = Array(UInt8, nrows, ncols, length(indicies))
    buffer = Array(UInt8, nrows, ncols)
    dst_index = 1
    for src_index in indicies
        readimage_raw!(buffer, io, src_index, nrows, ncols)
        copy!(images, 1 + nrows * ncols * (dst_index - 1), buffer, 1, nrows * ncols)
        dst_index += 1
    end
    images
end

"""
    readimage_raw(file, [indices])

Reads the images denoted by `indices` from `file`. The given `file`
can either be specified using an IO-stream or a string and is assumed
to be in the MNIST image-file format, as it is described on the
official homepage at http://yann.lecun.com/exdb/mnist/

- if `indices` is an `Integer`, the single image is returned as
`Matrix{UInt8}` in horizontal major layout, which means that the
first dimension denotes the pixel *rows* (x), and the second
dimension denotes the pixel *columns* (y) of the image.

- if `indices` is a `AbstractVector`, the images are returned as
a 3D tensor (i.e. a `Array{UInt8,3}`), in which the first dimension
corresponds to the pixel *rows* (x) of the image, the second
dimension to the pixel *columns* (y) of the image, and the third
dimension denotes the index of the image.

- if `indices` is ommited all images are returned
(as 3D Tensor described above)
"""
function readimage_raw(io::IO, indices)
    _, nimages, nrows, ncols = imageheader(io)
    @assert minimum(indices) >= 1 && maximum(indices) <= nimages
    readimage_raw(io, indices, nrows, ncols)
end

@noinline function readimage_raw(file::AbstractString, index::Integer)
    open(file, "r") do io
        readimage(io, index)
    end::Matrix{UInt8}
end

@noinline function readimage_raw(file::AbstractString, indices::AbstractVector)
    open(file, "r") do io
        readimage(io, indices)
    end::Array{UInt8,3}
end

@noinline function readimage_raw(file::AbstractString)
    open(file, "r") do io
        _, nimages, nrows, ncols = imageheader(io)
        readimage_raw(io, 1:nimages, nrows, ncols)
    end::Array{UInt8,3}
end


"""
    readimage(file::AbstractString, [indices])

Reads the images denoted by `indices` from `file`. The given `file`
is assumed to be in the MNIST image-file format, as it is described
on the official homepage at http://yann.lecun.com/exdb/mnist/

- if `indices` is an `Integer`, the single image is returned as
`Matrix{UInt8}` in vertical major layout, which means that the
first dimension denotes the pixel *columns* (y), and the second
dimension denotes the pixel *rows* (x) of the image.

- if `indices` is a `AbstractVector`, the images are returned as
a 3D tensor (i.e. a `Array{UInt8,3}`), in which the first dimension
corresponds to the pixel *columns* (y) of the image, the second
dimension to the pixel *rows* (x) of the image, and the third
dimension denotes the index of the image.

- if `indices` is ommited all images are returned
(as 3D Tensor described above)
"""
readimage(file) = permutedims(readimage_raw(file), [2,1,3])
readimage(file, index::Integer) = readimage_raw(file, index)'
readimage(file, index::AbstractVector) = permutedims(readimage_raw(file, index), [2,1,3])


# Horizontal major format (raw storing order)

imraw_doc(train_or_test) =
"""
    $(train_or_test)image_raw([T = UInt8], [indices])

Returns the MNIST **$(train_or_test)set** images denoted by `indices`.

- if `indices` is an `Integer`, the single image is returned as
`Matrix{T}` in horizontal major layout, which means that the first
dimension denotes the pixel *rows* (x), and the second dimension
denotes the pixel *columns* (y) of the image.

- if `indices` is a `AbstractVector`, the images are returned as
a 3D tensor (i.e. a `Array{T,3}`), in which the first dimension
corresponds to the pixel *rows* (x) of the image, the second
dimension to the pixel *columns* (y) of the image, and the third
dimension denotes the index of the image.

- if `indices` is ommited all images are returned
(as 3D Tensor described above)
"""

""" $(imraw_doc("train")) """
trainimage_raw() = readimage_raw(TRAINIMAGES)
trainimage_raw(indices) = readimage_raw(TRAINIMAGES, indices)
trainimage_raw{T}(::Type{T}) = Array{T}(trainimage_raw())
trainimage_raw{T}(::Type{T}, indices) = Array{T}(trainimage_raw(indices))

""" $(imraw_doc("test")) """
testimage_raw() = readimage_raw(TESTIMAGES)
testimage_raw(indices) = readimage_raw(TESTIMAGES, indices)
testimage_raw{T}(::Type{T}) = Array{T}(testimage_raw())
testimage_raw{T}(::Type{T}, indices) = Array{T}(testimage_raw(indices))


# Vertical major format (native julia convention)

im_doc(train_or_test) =
"""
    $(train_or_test)image([T = UInt8], [indices])

Returns the MNIST **$(train_or_test)set** images denoted by `indices`.

- if `indices` is an `Integer`, the single image is returned as
`Matrix{T}` in vertical major layout, which means that the first
dimension denotes the pixel *columns* (y), and the second dimension
denotes the pixel *rows* (x) of the image.

- if `indices` is a `AbstractVector`, the images are returned as
a 3D tensor (i.e. a `Array{T,3}`), in which the first dimension
corresponds to the pixel *columns* (y) of the image, the second
dimension to the pixel *rows* (x) of the image, and the third
dimension denotes the index of the image.

- if `indices` is ommited all images are returned
(as 3D Tensor described above)
"""

""" $(im_doc("train")) """
trainimage() = readimage(TRAINIMAGES)
trainimage(index) = readimage(TRAINIMAGES, index)
trainimage{T}(::Type{T}) = Array{T}(trainimage())
trainimage{T}(::Type{T}, index) = Array{T}(trainimage(index))

""" $(im_doc("test")) """
testimage() = readimage(TESTIMAGES)
testimage(index) = readimage(TESTIMAGES, index)
testimage{T}(::Type{T}) = Array{T}(testimage())
testimage{T}(::Type{T}, index) = Array{T}(testimage(index))

