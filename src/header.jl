"""
    imageheader(io::IO)

Reads four 32 bit integers at the beginning of `io`
and interprets them as a MNIST-image-file header, which
is described in detail in the table below

            ║     First    │  Second  │  Third  │   Fourth
    ════════╬══════════════╪══════════╪═════════╪════════════
    offset  ║         0000 │     0004 │    0008 │       0012
    descr   ║ magic number │ # images │  # rows │  # columns

These four numbers are returned as a Tuple in the same storage order
"""
function imageheader(io::IO)
    seekstart(io)
    magic_number = bswap(read(io, UInt32))
    total_items  = bswap(read(io, UInt32))
    nrows = bswap(read(io, UInt32))
    ncols = bswap(read(io, UInt32))
    magic_number, Int(total_items), Int(nrows), Int(ncols)
end


"""
    imageheader(file::AbstractString)

Opens and reads the first four 32 bits values of `file`
and returns them interpreted as an MNIST-image-file header
"""
function imageheader(file::AbstractString)
    open(imageheader, file, "r")
end


"""
    labelheader(io::IO)

Reads two 32 bit integers at the beginning of `io`
and interprets them as a MNIST-label-file header, which
consists of a *magic number* and the *total number of labels*
stored in the file. These two numbers are returned as a Tuple
in the same storage order.
"""
function labelheader(io::IO)
    seekstart(io)
    magic_number = bswap(read(io, UInt32))
    total_items  = bswap(read(io, UInt32))
    magic_number, Int(total_items)
end


"""
    labelheader(file::AbstractString)

Opens and reads the first two 32 bits values of `file`
and returns them interpreted as an MNIST-label-file header
"""
function labelheader(file::AbstractString)
    open(labelheader, file, "r")
end

