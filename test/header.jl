@test isfile(MNIST.TRAINIMAGES)
@test isfile(MNIST.TRAINLABELS)
@test isfile(MNIST.TESTIMAGES)
@test isfile(MNIST.TESTLABELS)

@test MNIST.IMAGEOFFSET == 16
@test MNIST.LABELOFFSET == 8

@test MNIST.imageheader(MNIST.TRAINIMAGES) == (0x803, 60000, 28, 28)
@test MNIST.imageheader(MNIST.TESTIMAGES)  == (0x803, 10000, 28, 28)

@test MNIST.labelheader(MNIST.TRAINLABELS) == (0x801, 60000)
@test MNIST.labelheader(MNIST.TESTLABELS)  == (0x801, 10000)

