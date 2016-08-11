# Check against the already tested image functions and make sure
# that the features are in appropriate formate
for (feature_fun, image_fun, nobs) in
        ((trainfeatures, MNIST.trainimage_raw, 60_000),
         (testfeatures,  MNIST.testimage_raw,  10_000))
    @testset "check $feature_fun against $image_fun" begin
        # Load all features
        A = feature_fun()
        @test typeof(A) <: Matrix{Float64}
        @test size(A) == (28*28,nobs)
        @test A == reshape(image_fun(Float64), (28*28,nobs))

        @testset "load single observations" begin
            # Sample a few random featurevectors to compare
            for i = rand(1:nobs, 200)
                A_i = feature_fun(i)
                @test typeof(A_i) <: Vector{Float64}
                @test size(A_i) == (28*28,)
                @test A_i == reshape(image_fun(Float64,i), 28*28)
            end
        end

        @testset "load multiple observations" begin
            A_5_10 = feature_fun(5:10)
            @test typeof(A_5_10) <: Matrix{Float64}
            @test size(A_5_10) == (28*28,6)
            for i = 1:6
                @test A_5_10[:,i] ==
                    reshape(image_fun(Float64, i+4), 28*28)
            end

            # also test edge cases `1`, `nobs`
            indices = [10,3,9,1,nobs]
            A_vec   = feature_fun(indices)
            A_vec_f = feature_fun(Vector{Int32}(indices))
            @test A_vec == A_vec_f
            @test typeof(A_vec) <: Matrix{Float64}
            @test size(A_vec) == (28*28,5)
            for i in 1:5
                @test A_vec[:,i] ==
                    reshape(image_fun(Float64, indices[i]), 28*28)
            end
        end
    end
end

# Check against the already tested feature and labels functions
for (data_fun, feature_fun, label_fun, nobs) in
        ((traindata, trainfeatures, trainlabel, 60_000),
         (testdata,  testfeatures,  testlabel,  10_000))
    @testset "check $data_fun against $feature_fun and $label_fun" begin
        data, labels = data_fun()
        @test data == feature_fun()
        @test labels == label_fun()

        for i = rand(1:nobs, 200)
            d_i, l_i = data_fun(i)
            @test d_i == feature_fun(i)
            @test l_i == label_fun(i)
        end

        data, labels = data_fun(5:10)
        @test data == feature_fun(5:10)
        @test labels == label_fun(5:10)

        indices = [10,3,9,1,nobs]
        data, labels = data_fun(indices)
        @test data == feature_fun(indices)
        @test labels == label_fun(indices)
    end
end

