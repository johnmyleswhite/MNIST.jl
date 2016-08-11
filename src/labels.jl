"""
    readlabel(io::IO, index::Integer)

Jumps to the position of `io` where the byte for the `index`'th
label is located and returns the byte at that position as `UInt8`
"""
function readlabel(io::IO, index::Integer)
    seek(io, LABELOFFSET + (index - 1))
    read(io, UInt8)::UInt8
end


"""
    readlabel(io::IO, indices::AbstractVector)

Reads the byte for each label-index in `indices` and stores them
in a `Vector{UInt8}` of length `length(indices)` in the same order
as denoted by `indices`.
"""
function readlabel(io::IO, indices::AbstractVector)
    labels = Array(UInt8, length(indices))
    dst_index = 1
    for src_index in indices
        labels[dst_index] = readlabel(io, src_index)
        dst_index += 1
    end
    labels::Vector{UInt8}
end


idx_doc(T) =
"""
- if `indices` is an `Integer`, the single label is returned as
`$(T)`.

- if `indices` is a `AbstractVector`, the labels are returned as
a `Vector{$(T)}`, length `length(indices)` in the same order as
denoted by `indices`.

- if `indices` is ommited all all are returned
(as `Vector{$(T)}` as described above)
"""

"""
    readlabel(file::AbstractString, [indices])

Reads the label denoted by `indices` from `file`. The given `file`
is assumed to be in the MNIST label-file format, as it is described
on the official homepage at http://yann.lecun.com/exdb/mnist/

$(idx_doc("UInt8"))
"""
@noinline function readlabel(file::AbstractString, index::Integer)
    open(file, "r") do io
        _, nlabels = labelheader(io)
        @assert minimum(index) >= 1 && maximum(index) <= nlabels
        readlabel(io, index)
    end::UInt8
end

@noinline function readlabel(file::AbstractString, indices::AbstractVector)
    open(file, "r") do io
        _, nlabels = labelheader(io)
        @assert minimum(indices) >= 1 && maximum(indices) <= nlabels
        readlabel(io, indices)
    end::Vector{UInt8}
end

@noinline function readlabel(file::AbstractString)
    open(file, "r") do io
        _, nlabels = labelheader(io)
        readlabel(io, 1:nlabels)
    end::Vector{UInt8}
end


# Public Interface

"""
    trainlabel([indices])

Returns the MNIST **trainset** labels denoted by `indices`.

$(idx_doc("Float64"))
"""
trainlabel() = Vector{Float64}(readlabel(TRAINLABELS))
trainlabel(index::Integer) = Float64(readlabel(TRAINLABELS, index))
trainlabel(indices::AbstractVector) = Vector{Float64}(readlabel(TRAINLABELS, indices))


"""
    testlabel([indices])

Returns the MNIST **testset** labels denoted by `indices`.

$(idx_doc("Float64"))
"""
testlabel() = Vector{Float64}(readlabel(TESTLABELS))
testlabel(index::Integer) = Float64(readlabel(TESTLABELS, index))
testlabel(indices::AbstractVector) = Vector{Float64}(readlabel(TESTLABELS, indices))

