include("ann.jl")

demo_ann(10,         # Number of training epochs
         0.1,        # Learning rate
         0.5,        # Momentum
         32,         # Size of mini-batches
         0.05,       # Standard deviation of Gaussian distributed initial weights
         [100, 100]) # Number of hidden nodes
