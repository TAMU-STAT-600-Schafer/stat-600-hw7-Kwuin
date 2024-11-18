# Function that implements multi-class logistic regression.
#############################################################
# Description of supplied parameters:
# X - n x p training data, 1st column should be 1s to account for intercept
# y - a vector of size n of class labels, from 0 to K-1
# Xt - ntest x p testing data, 1st column should be 1s to account for intercept
# yt - a vector of size ntest of test class labels, from 0 to K-1
# numIter - number of FIXED iterations of the algorithm, default value is 50
# eta - learning rate, default value is 0.1
# lambda - ridge parameter, default value is 1
# beta_init - (optional) initial starting values of beta for the algorithm, should be p x K matrix 


softmax <- function(X, beta) {
  scores <- X %*% beta # n x K
  # Subtract max for numerical stability
  scores <- sweep(scores, 1, apply(scores, 1, max))
  exp_scores <- exp(scores)
  probs <- sweep(exp_scores, 1, rowSums(exp_scores), "/")
  return(probs)
}

# Calculate objective function value
loss <- function(X, y, beta, lambda, K) {
  n <- nrow(X)
  probs <- softmax(X, beta)
  
  log_likelihood <- 0
  for (k in 0:(K - 1)) {
    class_idx <- which(y == k)
    if (length(class_idx) > 0) {
      log_likelihood <- log_likelihood + sum(log(probs[class_idx, k + 1]))
    }
  }
  
  ridge_penalty <- (lambda / 2) * sum(beta^2)
  
  return(-log_likelihood + ridge_penalty)
}


error <- function(X, y, beta) {
  probs <- softmax(X, beta)
  predictions <- max.col(probs) - 1 
  return(mean(predictions != y) * 100)
}



## Return output
##########################################################################
# beta - p x K matrix of estimated beta values after numIter iterations
# error_train - (numIter + 1) length vector of training error % at each iteration (+ starting value)
# error_test - (numIter + 1) length vector of testing error % at each iteration (+ starting value)
# objective - (numIter + 1) length vector of objective values of the function that we are minimizing at each iteration (+ starting value)
LRMultiClass <- function(X, y, Xt, yt, numIter = 50, eta = 0.1, lambda = 1, beta_init = NULL){
  ## Check the supplied parameters as described. You can assume that X, Xt are matrices; y, yt are vectors; and numIter, eta, lambda are scalars. You can assume that beta_init is either NULL (default) or a matrix.
  ###################################
  # Check that the first column of X and Xt are 1s, if not - display appropriate message and stop execution.
  if (!all(X[, 1] == 1)){
    stop("The first column of X must be 1s.")
  }
  # Check for compatibility of dimensions between X and Y
  if (nrow(X) != length(y)) {
    stop("X number of rows doesn't match length of Y")
  }
  # Check for compatibility of dimensions between Xt and Yt
  if (nrow(Xt) != length(yt)){
    stop("Xt number of rows doesn't match length of Yt")
  }
  # Check for compatibility of dimensions between X and Xt
  if (ncol(X) != ncol(Xt)){
    stop("X and Xt must have same number of columns")
  }
  # Check eta is positive
  if (eta <= 0){
    stop("eta must be positive")
  }
  # Check lambda is non-negative
  if (lambda < 0){
    stop("lambda must be non-negative")
  }
  # Check whether beta_init is NULL. If NULL, initialize beta with p x K matrix of zeroes. If not NULL, check for compatibility of dimensions with what has been already supplied.
  n <- nrow(X)
  p <- ncol(X)
  K <- length(unique(y))
  
  if (is.null(beta_init)) {
    beta <- matrix(0, nrow = p, ncol = K)
  } else {
    if (nrow(beta_init) != p || ncol(beta_init) != K) {
      stop("beta_init dimensions incompatible")
    }
    beta <- beta_init
  }
  ## Calculate corresponding pk, objective value f(beta_init), training error and testing error given the starting point beta_init
  ##########################################################################
  error_train <- numeric(numIter + 1)
  error_test <- numeric(numIter + 1)
  objective <- numeric(numIter + 1)
  
  objective[1] <- loss(X, y, beta, lambda, K)
  error_train[1] <- error(X, y, beta)
  error_test[1] <- error(Xt, yt, beta)
  ## Newton's method cycle - implement the update EXACTLY numIter iterations
  ##########################################################################
  for (iter in 1:numIter) {
    # Within one iteration: perform the update, calculate updated objective function and training/testing errors in %
    
    # Calculate probabilities
    probs <- softmax(X, beta)
    
    # Update each class
    for (k in 0:(K - 1)) {
      # Calculate W_k diagonal elements (without forming full matrix)
      w_k <- probs[, k + 1] * (1 - probs[, k + 1])
      
      # Efficient calculation of X'W_kX without forming W_k matrix
      XtWX <- t(X * w_k) %*% X
      
      # Create indicator vector
      y_k <- as.numeric(y == k)
      
      # Calculate gradient
      grad <- t(X) %*% (probs[, k + 1] - y_k) + lambda * beta[, k + 1]
      
      # Add ridge penalty to Hessian
      hessian <- XtWX + lambda * diag(p)
      
      # Update beta_k
      beta[, k + 1] <- beta[, k + 1] - eta * solve(hessian, grad)
    }
    
    # Store values
    objective[iter + 1] <- loss(X, y, beta, lambda, K)
    error_train[iter + 1] <- error(X, y, beta)
    error_test[iter + 1] <- error(Xt, yt, beta)
  }
  
  
  
  ## Return output
  ##########################################################################
  # beta - p x K matrix of estimated beta values after numIter iterations
  # error_train - (numIter + 1) length vector of training error % at each iteration (+ starting value)
  # error_test - (numIter + 1) length vector of testing error % at each iteration (+ starting value)
  # objective - (numIter + 1) length vector of objective values of the function that we are minimizing at each iteration (+ starting value)
  return(list(beta = beta, error_train = error_train, error_test = error_test, objective =  objective))
}