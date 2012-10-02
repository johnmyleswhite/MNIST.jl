# Working with the MNIST Data Set

The MNIST data set, available from http://yann.lecun.com/exdb/mnist/,
is stored in a binary format that is not always ideal for modeling.

By loading `utils.jl`, you gain access to six functions that make
working with the MNIST data easier:

* `read_train_image(index)`
* `read_test_image(index)`
* `read_train_label(index)`
* `read_test_label(index)`
* `read_train_features(index)`
* `read_test_features(index)`

For most people, it's easiest to use only four of these functions:

* `read_train_features(index)`
* `read_test_features(index)`
* `read_train_label(index)`
* `read_test_label(index)`

This will give you vectors of `Float64`'s as features and
individual `Float64`'s as labels.

# Install

* Download the data from Yann LeCun's site
* Place it in the `data` directory with the original filenames.

# Example Usage

    load("utils.jl")

    read_train_image(1)
    read_test_image(1)

    read_train_features(1)
    read_test_features(1)

    read_train_label(1)
    read_test_label(1)
