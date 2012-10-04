load("src/init.jl")

(magic_number, total_samples, nrows, ncols) = read_image_header("data/train-images.idx3-ubyte")

f = open("training.tsv", "w")
for index in 1:total_samples
  features = read_train_features(index)
  label = read_train_label(index)
  println(f, join([label, features], "\t"))
end
close(f)

(magic_number, total_samples, nrows, ncols) = read_image_header("data/t10k-images.idx3-ubyte")

f = open("test.tsv", "w")
for index in 1:total_samples
  features = read_test_features(index)
  label = read_test_label(index)
  println(f, join([label, features], "\t"))
end
close(f)
