# Overflow-tolerant inverse logit link function
function invlogit(z::Float64)
  if z < -100.0
    return 0.0
  elseif z > 100.0
    return 1.0
  else
    return 1.0 / (1.0 + exp(-z))
  end
end

# Estimate the performance of the model on a specific data set
function set_performance(w::Vector, target::Float64, filename::String)
  # Load metadata from data set
  (magic_number, total_samples, nrows, ncols) = read_image_header(filename)

  # Calculate error rate
  se = 0.0

  # Loop over entire training set
  for index in 1:total_samples
    # Extract features and label
    if filename == "data/train-images.idx3-ubyte"
      features = read_train_features(index)
      label = read_train_label(index)
    else
      features = read_test_features(index)
      label = read_test_label(index)
    end

    # Add an intercept
    x = vcat(1.0, features)

    # Define the truth value of y based on target label
    y = 0.0
    if label == target
      y = 1.0
    end

    # Calculate the prediction
    z = dot(w, x)
    h = invlogit(z)

    # Update accuracy
    se += (y - h)^2
  end

  # Report RMSE
  return sqrt(se / total_samples)
end

# Estimate performance on training set
function train_set_performance(w::Vector, target::Float64)
  set_performance(w, target, "data/train-images.idx3-ubyte")
end

# Estimate performance on test set
function test_set_performance(w::Vector, target::Float64)
  set_performance(w, target, "data/t10k-images.idx3-ubyte")
end

# Train the model
function train(w::Vector{Float64}, target::Float64, N::Int64, M::Int64, total_samples::Int64, eta::Float64, lambda::Float64)
  # Run for N iterations
  for iteration in 1:N
    # Select an image to predict
    index = randi(total_samples)

    # Load the data
    features = read_train_features(index)
    label = read_train_label(index)

    # Add an intercept
    x = vcat(1.0, features)

    # Define the truth value
    y = 0.0
    if label == target
      y = 1.0
    end

    # Calculate the prediction
    z = dot(w, x)
    h = invlogit(z)

    # Regularization step
    dw = (y - h) * x
    for j in 2:length(w)
      dw[j] += -lambda * w[j]
    end

    # Update the weights
    w += eta * dw

    # Report training and test set performance
    f = open("performance.csv", "a")
    if rem(iteration, M) == 0
      println(f, join({iteration, train_set_performance(w, target), test_set_performance(w, target), eta, lambda, norm(w)}, ","))
    end
    close(f)
  end
  return w
end
