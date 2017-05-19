# Sanity check that the first trainimage is not also the first testimage
@test MNIST.trainimage_raw(1) != MNIST.testimage_raw(1)
@test trainimage(1) != testimage(1)

# Make sure other integer types work as indicies
@test MNIST.trainimage_raw(0xBEEF) == MNIST.trainimage_raw(48879)
@test trainimage(0xBAE) == trainimage(2990)

# Sanity check that the images are returned transposed for the `_raw`
# methods. We don't check all the images but instead test a random
# sample to keep testing times reasonable.
# -- This doesn't actually check which one is the horizontal major
#    and which one is the vertical major. But it does make sure one
#    of them is the former and one of the the later!
@testset "Test that _raw is a transposed version" begin
    for i = rand(1:60_000, 200)
        @test MNIST.trainimage_raw(i)' == MNIST.trainimage(i)
    end
    for i = rand(1:10_000, 200)
        @test MNIST.testimage_raw(i)'  == MNIST.testimage(i)
    end
end

# Test that `_raw` is the horizontal-major layout by comparing to
# a hand checked result
@test MNIST.trainimage_raw(1)[11:13,12:14] ==
        [0x00 0x00 0x00;
         0x8b 0x0b 0x00;
         0xfd 0xbe 0x23]
@test MNIST.trainimage(1)[11:13,12:14] !=
        [0x00 0x00 0x00;
         0x8b 0x0b 0x00;
         0xfd 0xbe 0x23]

# These tests check if the functions return internaly consistent
# results for different parameters (e.g. index as int or as vector).
# That means no matter how you specify an index, you will always
# get the same result for a specific index.
# -- However, technically these tests do not check if these are the
#    actual MNIST images of that index!
for (image_fun, nimages, orientation) in
            ((MNIST.trainimage_raw, 60_000, "horizontal"),
             (MNIST.testimage_raw,  10_000, "horizontal"),
             (MNIST.trainimage,     60_000, "vertical"),
             (MNIST.testimage,      10_000, "vertical"))
    @testset "$image_fun: $orientation-major" begin
        # whole image set
        A   = image_fun()
        A_f = image_fun(Float32)
        @test typeof(A)   <: Array{UInt8,3}
        @test typeof(A_f) <: Array{Float32,3}
        @test size(A)   == (28,28,nimages)
        @test size(A_f) == (28,28,nimages)
        @test all(A .== A_f)

        @test_throws AssertionError image_fun(-1)
        @test_throws AssertionError image_fun(0)
        @test_throws AssertionError image_fun(nimages+1)

        @testset "load single images" begin
            # Sample a few random images to compare
            for i = rand(1:nimages, 200)
                A_i   = image_fun(i)
                A_i_f = image_fun(Float32, i)
                @test typeof(A_i)   <: Array{UInt8,2}
                @test typeof(A_i_f) <: Array{Float32,2}
                @test size(A_i)   == (28,28)
                @test size(A_i_f) == (28,28)
                @test all(A_i .== A_i_f)
                @test A_i == A[:,:,i]
            end
        end

        @testset "load multiple images" begin
            A_5_10   = image_fun(5:10)
            A_5_10_f = image_fun(Float32, 5:10)
            @test typeof(A_5_10)   <: Array{UInt8,3}
            @test typeof(A_5_10_f) <: Array{Float32,3}
            @test size(A_5_10)   == (28,28,6)
            @test size(A_5_10_f) == (28,28,6)
            for i = 1:6
                @test A_5_10[:,:,i] == A[:,:,i+4]
                @test A_5_10[:,:,i] == A_5_10_f[:,:,i]
            end

            # also test edge cases `1`, `nimages`
            indices = [10,3,9,1,nimages]
            A_vec   = image_fun(indices)
            A_vec_f = image_fun(Float32, Vector{Int32}(indices))
            @test typeof(A_vec)   <: Array{UInt8,3}
            @test typeof(A_vec_f) <: Array{Float32,3}
            @test size(A_vec)   == (28,28,5)
            @test size(A_vec_f) == (28,28,5)
            for i in 1:5
                @test A_vec[:,:,i] == A[:,:,indices[i]]
                @test A_vec[:,:,i] == A_vec_f[:,:,i]
            end
        end
    end
end

