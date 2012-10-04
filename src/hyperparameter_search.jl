# Load utility functions
load("src/init.jl")

# Load metadata from training set
(magic_number, total_samples, nrows, ncols) = read_image_header("data/train-images.idx3-ubyte")

# Total number of training steps
N = 2_500

# Report every M iterations
M = 25

# Set the target label
target = 4.0

# Drop the previous training results
file_remove("performance.csv")

# Loop over hyperparameters
for lambda in [0.0, 0.1, 1.0, 10.0]
  for eta in [10e-6, 10e-8, 10e-10, 10e-12]
    # Initialize zero weights
    w = zeros(nrows * ncols + 1)

    # Train the algorithm
    w = train(w, target, N, M, total_samples, eta, lambda)
  end
end
