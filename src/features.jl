_flattenimage(image::Matrix) = vec(image)

function _flattenimage{T}(images::Array{T,3})
    ncols, nrows, nimages = size(images)
    reshape(images, (nrows * ncols, nimages))
end


feature_doc(train_or_test) =
"""
    $(train_or_test)features([indices])

Returns the MNIST **$(train_or_test)set** features denoted by `indices`.

- if `indices` is an `Integer`, the single image is returned as
flat `Vector{Float64}` feature vector.

- if `indices` is a `AbstractVector`, the images are returned as
a design matrix `Matrix{Float64}` in feature major layout, which
means that the first dimension denotes the features (independent
variables) and the second dimension denotes the observations (images)

- if `indices` is ommited all images are returned
(as a design matrix described above)
"""

""" $(feature_doc("train")) """
trainfeatures() = _flattenimage(trainimage_raw(Float64))
trainfeatures(index) = _flattenimage(trainimage_raw(Float64, index))

""" $(feature_doc("test")) """
testfeatures() = _flattenimage(testimage_raw(Float64))
testfeatures(index) = _flattenimage(testimage_raw(Float64, index))



data_doc(train_or_test) =
"""
    $(train_or_test)data([indices])

Returns the MNIST **$(train_or_test)set** features and labels,
denoted by `indices`, as Tuple.

- if `indices` is an `Integer`, the single observation is returned
as a `Tuple{Vector{Float64}, Float64}`, where the first element
is a plain feature vector, and the second element is the label of
the image.

- if `indices` is a `AbstractVector`, the observations are returned
as a `Tuple{Matrix{Float64}, Vector{Float64}}`.
The the first element of the tuple is a design matrix
`Matrix{Float64}` in feature major layout, which means that its
first dimension denotes the features and its second dimension
denotes the observations (images).
The second element of the Tuple is a vector of labels for each
image in the design matrix (in the same order)

- if `indices` is ommited all observations are returned
(as a Tuple of design matrix and label-vector described above)
"""

""" $(data_doc("train")) """
traindata() = trainfeatures(), trainlabel()
traindata(index) = trainfeatures(index), trainlabel(index)

""" $(data_doc("test")) """
testdata() = testfeatures(), testlabel()
testdata(index) = testfeatures(index), testlabel(index)

