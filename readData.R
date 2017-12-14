setwd('C:/Users/Asus/Documents/Project')
#install.packages("rjson")
library("rjson")
data <- fromJSON(file = "final_data.json")

### DATA PROCESSING
# Extracting the number of blue, red and green samples.
# N[1]: blues, N[2]: reds, N[3]: greens
N <- unname(sapply(data[[1]],'[[',2))

# Generating the matrix samples X, vector Y (where 0: blue, 1: red, 2: green) 
X <- c()
for (i in 1:length(data[[2]])){
  X <- rbind(X, as.vector(apply(as.matrix(data[[2]][[i]][[4]]), 1, function(x) unlist(x)))) #ASK!!!! t(M) if by row
}

Y <- matrix(c(rep(c(0,0,1),N[1]),rep(c(0,1,0),N[2]),rep(c(1,0,0),N[3])), ncol = 3, byrow = TRUE)
#Y <- c(rep(0,N[1]),rep(1,N[2]),rep(2,N[3]))

# Pretreatment of the data
# [sigmamin,sigmamax] this is [0,1] for the logistic function
sigma_min <- 0
sigma_max <- 1

#Y_max <- max(Y)
#Y_min <- min(Y)
#Y_treat <- (sigma_max-sigma_min)/(Y_max-Y_min)*(Y-Y_min)+sigma_min

X_max <- apply(X,2,max)
X_min <- apply(X,2,min)
X_treat <- (sigma_max-sigma_min)/(X_max-X_min)*(X-X_min)+sigma_min
#intento2 <- c()
#for (i in 1:ncol(X)){
#  intento2 <- cbind(intento2, (sigma_max-sigma_min)/(X_max[i]-X_min[i])*(X[,i]-X_min[i])+sigma_min)
#}
#all.equal(X_treat,intento2)


####################################################################################################

### NEURONAL NETWORK
# Defining the activation function
sigmoid <- function(z){
  return(1/(1 + exp(-z)))
}

forward <- function(X, Weights, length_l1, p, k){
  # Weights
  Theta1 <- matrix(Weights[1:(length_l1*(p+1))],nrow=length_l1)
  Theta2 <- matrix(Weights[((length_l1*(p+1))+1):length(Weights)],nrow=k)
  
  # Including intercept in X and transposing the matrix
  n <- nrow(X)
  X_NN <- cbind(rep(1,n),X)
  
  # Computaring neurons values and output
  l1 <- sigmoid(X_NN%*%t(Theta1))
  l1_i <- cbind(rep(1,n),l1)
  l2 <- sigmoid(l1_i%*%t(Theta2))
  
  M <- apply(l2,1,max)
  Y_classified <- floor(l2/M)
  #Y_classified <- apply(sigmoid(ThetaF%*%l1_i),2,FUN=which.max)-1
  #Empirical_error_NN <- length(which(Y_classified!=Y))/n
  
  return(list( Theta1=Theta1, Theta2=Theta2, l1=l1, l2= l2, Y_classified=Y_classified))
}


## BACKWARD PROPAGATION
nnminuslogLikelihood <- function(Weights, p, length_l1, k, X, Y, lambda){
  n <- dim(X)[1]
  
  # Forward step
  forward_result <- forward(X, Weights, length_l1, p, k)
  Theta1 <- forward_result$Theta1
  Theta2 <- forward_result$Theta2
  l2 <- forward_result$l2 
  
  # Computing the log-likelihood
  J <- 0
  J <- apply(-Y*log(l2),1,sum)-apply((1-Y)*log(1-l2),1,sum)
  J <- sum(J)/n+(lambda/(2*n))*(sum(Theta1[,2:dim(Theta1)[2]]^2)+sum(Theta2[,2:dim(Theta2)[2]]^2))
  return(J)
}


nnminuslogLikelihood_grad <- function(Weights, p, length_l1, k, X, Y, lambda){
  n <- dim(X)[1]
  
  # Forward step
  forward_result <- forward(X, Weights, length_l1, p, k)
  Theta1 <- forward_result$Theta1
  Theta2 <- forward_result$Theta2
  l1 <- forward_result$l1
  l2 <- forward_result$l2 
  
  delta3 <- t(l2-Y)
  Delta2 <- delta3%*%cbind(rep(1,n),l1)
  delta2 <- (t(Theta2)%*%delta3)[2:(length_l1+1),]*(t(l1)*t(1-l1))
  Delta1 <- delta2%*%cbind(rep(1,n),X)
  
  Theta1_grad = Delta1/n
  Theta1_grad[, 2:dim(Theta1_grad)[2]] = Theta1_grad[, 2:dim(Theta1_grad)[2]] + (lambda/n) * Theta1[ ,2:dim(Theta1_grad)[2]]
  Theta2_grad = Delta2/n
  Theta2_grad[, 2:dim(Theta2_grad)[2]] = Theta2_grad[, 2:dim(Theta2_grad)[2]] + (lambda/n) * Theta2[ ,2:dim(Theta2_grad)[2]]
  
  grad = c(as.vector(Theta1_grad), as.vector(Theta2_grad))
}

## FORWARD PROPAGATION
# Definig weights and number of neurons
p <- ncol(X)
k <- length(N)  #number of classes
set.seed(12345)
TrainingSize <- floor(0.6*nrow(X))
SelectRow <- sample(seq_len(nrow(X)), size = TrainingSize) 
TrainingData <- X_treat[SelectRow, ]
TrainingOutput <- Y[SelectRow, ]
ValidationData <- X_treat[-SelectRow, ]
ValidationOutput <- Y[-SelectRow, ]
MSE_Training <- c()
MSE_Testing <- c()



for (i in 5:50){
  Theta1 <- matrix(runif(i*(p+1)),nrow=i)
  ThetaF <- matrix(runif(k*(i+1)),nrow=k)
  Weights <- c(as.vector(Theta1),as.vector(ThetaF))

  #options = list(trace=1, iter.max=100) # print every iteration 
  backprop_result = nlminb(Weights, objective=nnminuslogLikelihood, 
                      gradient = nnminuslogLikelihood_grad, hessian = NULL,
                      p=p, length_l1=i, k=k, 
                      X=TrainingData, Y=TrainingOutput, lambda=1)
                      #control = options)
  #nn_params_backprop = backprop_result$par
  #cost = backprop_result$objective
  
  Weights_backp <- backprop_result$par

  Y_Train <- forward(TrainingData, Weights_backp, i, p, k)$l2
  Y_Test <- forward(ValidationData, Weights_backp, i, p, k)$l2
  MSE_Training <- c(MSE_Training,sum((Y_Train-TrainingOutput)^2)/nrow(TrainingOutput))
  MSE_Testing <- c(MSE_Testing,sum((Y_Test-ValidationOutput)^2)/nrow(ValidationOutput))
}
