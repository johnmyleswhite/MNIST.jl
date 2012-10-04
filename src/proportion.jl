load("src/init.jl")

(magic_number, total_samples, nrows, ncols) = read_image_header("data/t10k-images.idx3-ubyte")

occurences = Dict{Float64, Int64}()

for label in 0.0:9.0
  occurences[label] = 0
end

for index in 1:total_samples
  label = read_test_label(index)
  occurences[label] += 1
end

for label in 0.0:9.0
  println(join({label, occurences[label] / total_samples}, "\t"))
end
