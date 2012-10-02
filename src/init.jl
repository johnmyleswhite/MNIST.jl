function read_image_header(filename::String)
  f = open(filename, "r")
  magic_number = bswap(read(f, Uint32))
  total_items = bswap(read(f, Uint32))
  nrows = bswap(read(f, Uint32))
  ncols = bswap(read(f, Uint32))
  close(f)
  (magic_number, int(total_items), int(nrows), int(ncols))
end

function read_image(filename::String, index::Int64)
  f = open(filename, "r")
  seek(f, 16 + 28 * 28 * (index - 1))
  image = zeros(Uint8, 28, 28)
  for i in 1:28
    for j in 1:28
      image[i, j] = read(f, Uint8)
    end
  end
  close(f)
  return image
end

function read_label_header(filename::String)
  f = open(filename, "r")
  magic_number = bswap(read(f, Uint32))
  total_items = bswap(read(f, Uint32))
  close(f)
  return (magic_number, int(total_items))
end

function read_label(filename::String, index::Int64)
  f = open(filename, "r")
  seek(f, 8 + (index - 1))
  label = read(f, Uint8)
  close(f)
  return label
end

function read_train_image(index::Int64)
  convert(Array{Float64,2},
          read_image(file_path("data",
                               "train-images.idx3-ubyte"),
                     index))
end

function read_test_image(index::Int64)
  convert(Array{Float64,2},
          read_image(file_path("data",
                               "t10k-images.idx3-ubyte"),
                     index))
end

function read_train_label(index::Int64)
  convert(Float64,
          read_label(file_path("data",
                               "train-labels.idx1-ubyte"),
                     index))
end

function read_test_label(index::Int64)
  convert(Float64,
          read_label(file_path("data",
                               "t10k-labels.idx1-ubyte"),
                     index))
end

read_train_features(index::Int64) = reshape(read_train_image(index), 28*28)
read_test_features(index::Int64) = reshape(read_train_image(index), 28*28)
