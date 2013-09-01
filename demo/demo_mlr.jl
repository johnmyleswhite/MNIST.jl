include("mlr.jl")

demo_mlr(10,  # Number of training epochs
         0.1, # Learning rate
         0.5, # Momentum
         32)  # Size of mini-batches
