load("utils.jl")

filename = file_path("data", "train-images.idx3-ubyte")

f = open("images.tsv", "w")
for index in 1:10
  image = read_image(filename, index)
  for i in 1:28
    println(f, join(map(e -> int(e), image[i, 1:28]), "\t"))
  end
end
close(f)

filename = file_path("data", "train-labels.idx1-ubyte")

f = open("labels.tsv", "w")
for index in 1:10
  label = read_label(filename, index)
  println(f, read_label(filename, index))
end
close(f)

read_train_image(1)
read_test_image(1)
read_train_label(1)
read_test_label(1)

read_train_features(1)
read_test_features(1)
