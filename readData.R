setwd('C:/Users/Asus/Documents/Project')
#install.packages("rjson")
library("rjson")
data <- fromJSON(file = "final_data.json")

## DATA PROCESSING
# Extracting the number of blue, red and green samples.
# N[1]: blues, N[2]: reds, N[3]: greens
N <- unname(sapply(data[[1]],'[[',2))

# Generating the matrix samples X, vector Y (where 0: blue, 1: red, 2: green) 
X <- c()
for (i in 1:length(data[[2]])){
  X <- rbind(X, as.vector(apply(as.matrix(data[[2]][[i]][[4]]), 1, function(x) unlist(x)))) #ASK!!!! t(M) if by row
}

Y <- matrix(c(rep(c(0,0,1),N[1]),rep(c(0,1,0),N[2]),rep(c(1,0,0),N[3])), ncol = 3, byrow = TRUE)
# Y <- c(rep(0,N[1]),rep(1,N[2]),rep(2,N[3]))

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

## NEURONAL NETWORK
# Defining the activation function
sigmoid <- function(z){
  return(1/(1 + exp(-z)))
}

# FORWARD PROPAGATION
# Including intercept in X and transposing the matrix
n <- nrow(X)
p <- ncol(X)
k <- length(N)  #number of classes
X_NN <- t(cbind(rep(1,n),X_treat))

# Definig weights and number of neurons
lenght_l1 <- 25
Theta1 <- matrix(runif(lenght_l1*(p+1)),nrow=lenght_l1)
ThetaF <- matrix(runif(k*(lenght_l1+1)),nrow=k)

# Computaring neurons values and output
l1 <- sigmoid(Theta1%*%X_NN)
l1_i <- rbind(rep(1,n),l1)

Y_classified <- sigmoid(ThetaF%*%l1_i)
Y_classified[which(apply(Y_classified, 2, function(x) x != max(x,na.rm=TRUE)))] <- 0
Y_classified[which(apply(Y_classified, 2, function(x) x != 0))] <- 1
# Y_classified <- apply(sigmoid(ThetaF%*%l1_i),2,FUN=which.max)
# Empirical_error_NN <- length(which(Y_classified!=Y))/n


## BACKWARD PROPAGATION
# Loss function
