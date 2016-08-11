# Sanity check that the first trainlabel is not also the first testlabel
@test trainlabel(1) != testlabel(1)

# Check a few hand picked examples. I looked at both the pictures and
# the native output to make sure these values are correspond to the
# image at the same index.
@test trainlabel(1) == 5.
@test trainlabel(2) == 0.
@test trainlabel(1337) == 3.
@test trainlabel(0xCAFE) == 6.
@test trainlabel(60_000) == 8.
@test testlabel(1) == 7.
@test testlabel(2) == 2.
@test testlabel(0xDAD) == 4.
@test testlabel(10_000) == 6.

# These tests check if the functions return internaly consistent
# results for different parameters (e.g. index as int or as vector).
# That means no matter how you specify an index, you will always
# get the same result for a specific index.
# -- However, technically these tests do not check if these are the
#    actual MNIST labels of that index!
for (label_fun, nlabels) in
            ((trainlabel, 60_000),
             (testlabel,  10_000))
    @testset "$label_fun" begin
        # whole label set
        A = label_fun()
        @test typeof(A) <: Vector{Float64}
        @test size(A) == (nlabels,)

        @testset "load single label" begin
            # Sample a few random labels to compare
            for i = rand(1:nlabels, 200)
                A_i = label_fun(i)
                @test typeof(A_i) <: Float64
                @test A_i == A[i]
            end
        end

        @testset "load multiple labels" begin
            A_5_10 = label_fun(5:10)
            @test typeof(A_5_10) <: Vector{Float64}
            @test size(A_5_10) == (6,)
            for i = 1:6
                @test A_5_10[i] == A[i+4]
            end

            # also test edge cases `1`, `nlabels`
            indices = [10,3,9,1,nlabels]
            A_vec   = label_fun(indices)
            A_vec_f = label_fun(Vector{Int32}(indices))
            @test typeof(A_vec)   <: Vector{Float64}
            @test typeof(A_vec_f) <: Vector{Float64}
            @test size(A_vec)   == (5,)
            @test size(A_vec_f) == (5,)
            for i in 1:5
                @test A_vec[i] == A[indices[i]]
                @test A_vec[i] == A_vec_f[i]
            end
        end
    end
end

