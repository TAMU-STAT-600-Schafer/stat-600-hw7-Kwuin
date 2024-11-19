# Initialization
#####################################################
# p - dimension of input layer
# hidden_p - dimension of hidden layer
# K - number of classes, dimension of output layer
# scale - magnitude for initialization of W_k (standard deviation of normal)
# seed - specified seed to use before random normal draws
initialize_bw <- function(p, hidden_p, K, scale = 1e-3, seed = 12345){
  # [ToDo] Initialize intercepts as zeros
  
  set.seed(seed)
  
  # Initialize intercepts as zeros
  b1 <- rep(0, hidden_p)  # hidden layer intercepts
  b2 <- rep(0, K)        # output layer intercepts
  
  # Initialize weights with normal distribution
  W1 <- matrix(rnorm(p * hidden_p, mean = 0, sd = scale), nrow = p)    # input to hidden
  W2 <- matrix(rnorm(hidden_p * K, mean = 0, sd = scale), nrow = hidden_p)  # hidden to output
  
  return(list(b1 = b1, b2 = b2, W1 = W1, W2 = W2))
}

# Function to calculate loss, error, and gradient strictly based on scores
# with lambda = 0
#############################################################
# scores - a matrix of size n by K of scores (output layer)
# y - a vector of size n of class labels, from 0 to K-1
# K - number of classes
loss_grad_scores <- function(y, scores, K){
  #browser()
  n = length(y)
  # [ToDo] Calculate loss when lambda = 0
  scores_shifted <- sweep(scores, 1, apply(scores, 1, max))  # for numerical stability
  exp_scores <- exp(scores_shifted)
  probs <- sweep(exp_scores, 1, rowSums(exp_scores), "/")
  if (any(is.na(probs)) || any(probs < 0) || any(probs > 1)) {
    stop("Invalid probabilities calculated")
  }
  correct_probs <- probs[cbind(1:n, y + 1)]
  if (any(correct_probs == 0)) {
    correct_probs[correct_probs == 0] <- .Machine$double.eps  # prevent log(0)
  }
  loss <- -mean(log(correct_probs))
  
  # [ToDo] Calculate misclassification error rate (%)
  # when predicting class labels using scores versus true y
  predictions <- max.col(scores, ties.method = "random") - 1  # -1 to make 0-based
  error <- mean(predictions != y) * 100
  
  
  # [ToDo] Calculate gradient of loss with respect to scores (output)
  grad <- probs
  grad[cbind(1:n, y + 1)] <- grad[cbind(1:n, y + 1)] - 1
  grad <- grad / n
  
  # Return loss, gradient and misclassification error on training (in %)
  return(list(loss = loss, grad = grad, error = error))
}


# One pass function
################################################
# X - a matrix of size n by p (input)
# y - a vector of size n of class labels, from 0 to K-1
# W1 - a p by h matrix of weights
# b1 - a vector of size h of intercepts
# W2 - a h by K matrix of weights
# b2 - a vector of size K of intercepts
# lambda - a non-negative scalar, ridge parameter for gradient calculations
one_pass <- function(X, y, K, W1, b1, W2, b2, lambda) {
  n <- nrow(X)
  #browser()
  # Forward pass
  # From input to hidden
  # Correct broadcasting of b1
  hidden_raw <- X %*% W1 + matrix(rep(b1, each = n), nrow = n)

  # ReLU activation
  hidden <- pmax(hidden_raw,0)
  #browser()
  # From hidden to output scores
  # Correct broadcasting of b2
  scores <- hidden %*% W2 + matrix(rep(b2, each = n), nrow = n)
  
  # Get loss, error, gradient at current scores
  out <- loss_grad_scores(y, scores, K)
  
  # Backward pass
  # Gradient for W2, b2
  dscores <- out$grad
  dW2 <- t(hidden) %*% dscores + lambda * W2
  db2 <- colSums(dscores)
  
  # Gradient for hidden layer
  dhidden <- dscores %*% t(W2)
  dhidden[hidden_raw <= 0] <- 0  # ReLU gradient
  
  # Gradient for W1, b1
  dW1 <- t(X) %*% dhidden + lambda * W1
  db1 <- colSums(dhidden)
  
  return(list(
    loss = out$loss,
    error = out$error,
    grads = list(dW1 = dW1, db1 = db1, dW2 = dW2, db2 = db2)
  ))
}

# Function to evaluate validation set error
####################################################
# Xval - a matrix of size nval by p (input)
# yval - a vector of size nval of class labels, from 0 to K-1
# W1 - a p by h matrix of weights
# b1 - a vector of size h of intercepts
# W2 - a h by K matrix of weights
# b2 - a vector of size K of intercepts
evaluate_error <- function(Xval, yval, W1, b1, W2, b2){
  n <- nrow(Xval)
  
  # [ToDo] Forward pass to get scores on validation data
  hidden <- pmax(Xval %*% W1 + rep(1, n) %*% t(b1), 0)  # With ReLU
  scores <- hidden %*% W2 + rep(1, n) %*% t(b2)
  
  # [ToDo] Evaluate error rate (in %) when 
  # comparing scores-based predictions with true yval
  predictions <- max.col(scores) - 1  # -1 to make 0-based
  error <- mean(predictions != yval) * 100
  
  return(error)
}


# Full training
################################################
# X - n by p training data
# y - a vector of size n of class labels, from 0 to K-1
# Xval - nval by p validation data
# yval - a vector of size nval of of class labels, from 0 to K-1, for validation data
# lambda - a non-negative scalar corresponding to ridge parameter
# rate - learning rate for gradient descent
# mbatch - size of the batch for SGD
# nEpoch - total number of epochs for training
# hidden_p - size of hidden layer
# scale - a scalar for weights initialization
# seed - for reproducibility of SGD and initialization
NN_train <- function(X, y, Xval, yval, lambda = 0.01,
                     rate = 0.01, mbatch = 20, nEpoch = 100,
                     hidden_p = 20, scale = 1e-3, seed = 12345) {
  # Set seed for reproducibility
  set.seed(seed)
  
  # Get dimensions
  n <- length(y)
  p <- ncol(X)
  K <- length(unique(y))
  nBatch <- floor(n/mbatch)
  
  params <- initialize_bw(p, hidden_p, K, scale, seed)
  W1 <- params$W1
  b1 <- params$b1
  W2 <- params$W2
  b2 <- params$b2
  
  # Initialize storage
  error <- rep(NA, nEpoch)
  error_val <- rep(NA, nEpoch)
  
  initial_pass <- one_pass(X, y, K, W1, b1, W2, b2, lambda)
  cat("Initial loss:", initial_pass$loss, "Initial error:", initial_pass$error, "\n")
  
  for (i in 1:nEpoch) {
    batch_idx <- sample(n)
    batch_errors <- numeric(nBatch)
    
    for(j in 1:nBatch) {
      # Get current batch
      current_idx <- batch_idx[((j-1)*mbatch + 1):min(j*mbatch, n)]
      
      # Forward and backward pass
      out <- one_pass(X[current_idx,,drop=FALSE], y[current_idx], K, W1, b1, W2, b2, lambda)
      
      # Store batch error
      batch_errors[j] <- out$error
      
      # Print gradients norm to debug
      if(i == 1 && j == 1) {
        cat("First batch gradients norm:",
            sqrt(sum(out$grads$dW1^2)),
            sqrt(sum(out$grads$dW2^2)), "\n")
      }
      
      # Simple SGD update
      W1 <- W1 - rate * out$grads$dW1
      b1 <- b1 - rate * out$grads$db1
      W2 <- W2 - rate * out$grads$dW2
      b2 <- b2 - rate * out$grads$db2
    }
    
    # Calculate and store errors
    full_pass <- one_pass(X, y, K, W1, b1, W2, b2, lambda)
    error[i] <- mean(batch_errors)
    error_val[i] <- evaluate_error(Xval, yval, W1, b1, W2, b2)
    
    # Print every 10 epochs
    if(i %% 10 == 0) {
      cat("Epoch", i, "Loss:", error[i], "Val error:", error_val[i], "\n")
    }
  }
  
  return(list(error = error,
              error_val = error_val,
              params = list(W1 = W1, b1 = b1, W2 = W2, b2 = b2)))
}