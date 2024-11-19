# Test file for Neural Network Implementation
library(testthat)

# Test 1: Initialization Function
test_that("initialize_bw produces correct dimensions and values", {
  source("FunctionsNN.R")
  set.seed(12345)
  p <- 3        # input dimension
  hidden_p <- 4 # hidden layer dimension
  K <- 2        # number of classes
  
  result <- initialize_bw(p, hidden_p, K)
  
  # Check dimensions
  expect_equal(length(result$b1), hidden_p)
  expect_equal(length(result$b2), K)
  expect_equal(dim(result$W1), c(p, hidden_p))
  expect_equal(dim(result$W2), c(hidden_p, K))
  
  # Check if biases are initialized to zero
  expect_equal(sum(abs(result$b1)), 0)
  expect_equal(sum(abs(result$b2)), 0)
  
  # Check if weights are small (using scale = 1e-3)
  expect_true(max(abs(result$W1)) < 0.01)
  expect_true(max(abs(result$W2)) < 0.01)
})

# Test 2: Loss and Gradient Function with Simple Cases
test_that("loss_grad_scores works correctly for simple cases", {
  set.seed(123)
  source("FunctionsNN.R")
  # Case 1: Perfect prediction
  y <- c(0, 1)
  scores <- matrix(c(100, -100, -100, 100), nrow=2)  # Strong predictions
  result <- loss_grad_scores(y, scores, K=2)
  
  # Loss should be very small for perfect predictions
  expect_true(result$loss < 0.01)
  # Error should be 0
  expect_equal(result$error, 0)
  
  # Case 2: Complete uncertainty
  scores2 <- matrix(c(0, 0, 0, 0), nrow=2)  # Equal scores
  result2 <- loss_grad_scores(y, scores2, K=2)
  
  # Loss should be around log(2) for binary case with equal probabilities
  expect_true(abs(result2$loss - log(2)) < 0.01)
  # Error should be 50% for random guessing in binary case
  
  #expect_true(abs(result2$error - 50) < 0.01)
})

# Test 3: Test with Two Normal Populations
test_that("NN can learn to separate two normal populations", {
  source("FunctionsNN.R")
  set.seed(12345)
  
  # Generate two normal populations
  n_per_class <- 100
  X1 <- matrix(rnorm(n_per_class*2, mean=-2), ncol=2)
  X2 <- matrix(rnorm(n_per_class*2, mean=2), ncol=2)
  X <- rbind(X1, X2)
  X <- cbind(1, X)  # Add intercept
  y <- c(rep(0, n_per_class), rep(1, n_per_class))
  
  # Generate validation data
  X1_val <- matrix(rnorm(20*2, mean=-2), ncol=2)
  X2_val <- matrix(rnorm(20*2, mean=2), ncol=2)
  Xval <- rbind(X1_val, X2_val)
  Xval <- cbind(1, Xval)
  yval <- c(rep(0, 20), rep(1, 20))
  
  # Train network
  source("FunctionsNN.R")
  result <- NN_train(X, y, Xval, yval, 
                     lambda=0.01, rate=0.01, 
                     mbatch=20, nEpoch=50,
                     hidden_p=10)
  
  # Check if error decreases
  expect_true(result$error[length(result$error)] < result$error[1])
  expect_true(result$error_val[length(result$error_val)] < result$error_val[1])
  
  # Final error should be reasonably low for well-separated populations
  expect_true(result$error[length(result$error)] < 10)
})

# Test 4: Test One Pass Function
test_that("one_pass produces correct output format and reasonable values", {
  set.seed(12345)
  
  # Small test case
  n <- 10
  p <- 3
  hidden_p <- 4
  K <- 2
  
  # Generate simple data
  X <- matrix(rnorm(n*p), ncol=p)
  y <- sample(0:(K-1), n, replace=TRUE)
  
  # Initialize parameters
  params <- initialize_bw(p, hidden_p, K)
  
  # Run one pass
  result <- one_pass(X, y, K, 
                     params$W1, params$b1, 
                     params$W2, params$b2, 
                     lambda=0.01)
  
  # Check output format
  expect_true(is.list(result))
  expect_true(all(c("loss", "error", "grads") %in% names(result)))
  expect_true(all(c("dW1", "db1", "dW2", "db2") %in% names(result$grads)))
  
  # Check dimensions of gradients
  expect_equal(dim(result$grads$dW1), dim(params$W1))
  expect_equal(dim(result$grads$dW2), dim(params$W2))
  expect_equal(length(result$grads$db1), length(params$b1))
  expect_equal(length(result$grads$db2), length(params$b2))
  
  # Check if loss and error are reasonable
  expect_true(result$loss > 0)
  expect_true(result$error >= 0 && result$error <= 100)
})

# Test 5: Check Learning Process
test_that("Network shows proper learning behavior", {
  set.seed(12345)
  
  # Generate linearly separable data
  n <- 200
  
  # First class: points in first and third quadrants
  X0 <- rbind(
    matrix(runif(n/4 * 2, 0, 3), ncol=2),    # first quadrant
    matrix(runif(n/4 * 2, -3, 0), ncol=2)     # third quadrant
  )
  
  # Second class: points in second and fourth quadrants
  X1 <- rbind(
    matrix(c(runif(n/4, -3, 0), runif(n/4, 0, 3)), ncol=2),    # second quadrant
    matrix(c(runif(n/4, 0, 3), runif(n/4, -3, 0)), ncol=2)     # fourth quadrant
  )
  
  # Combine data
  X <- rbind(X0, X1)
  X <- cbind(1, X)  # Add intercept
  y <- c(rep(0, n/2), rep(1, n/2))
  
  # Optional: visualize to verify separability
  # plot(X[,2], X[,3], col=y+1, pch=19)
  
  # Split into train and validation
  train_idx <- sample(n, 0.8*n)
  X_train <- X[train_idx,]
  y_train <- y[train_idx]
  X_val <- X[-train_idx,]
  y_val <- y[-train_idx]
  
  # Train with different learning rates
  result1 <- NN_train(X_train, y_train, X_val, y_val, 
                      rate=0.3, nEpoch=30)
  result2 <- NN_train(X_train, y_train, X_val, y_val, 
                      rate=0.1, nEpoch=30)
  
  # Check learning properties
  # 1. Error should decrease
  expect_true(diff(result1$error)[1] < 0)
  
  # 2. Final error should be better than initial
  expect_true(tail(result1$error, 1) < result1$error[1])
  
  # 3. Higher learning rate should lead to faster initial improvement
  early_improvement1 <- result1$error[5] - result1$error[1]
  early_improvement2 <- result2$error[5] - result2$error[1]
  expect_true(abs(early_improvement2) < abs(early_improvement1))
  
  # 4. Check for overfitting
  train_vs_val_correlation <- cor(result1$error, result1$error_val)
  expect_true(train_vs_val_correlation > 0.5)
})
