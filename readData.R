setwd('C:/Users/Asus/Documents/Project')
#install.packages("rjson")
library("rjson")
data <- fromJSON(file = "final_data.json")


# Extracting the number of blue, red and green samples.
# N[1]: blues, N[2]: reds, N[3]: greens
N <- unname(sapply(data[[1]],'[[',2))


# Generating the matrix samples X, vector Y (where 0: blue, 1: red, 2: green) 
X <- c()
for (i in 1:length(data[[2]])){
  X <- rbind(X, as.vector(apply(as.matrix(data[[2]][[i]][[4]]), 1, function(x) unlist(x)))) #ASK!!!! t(M) if by row
}

Y <- c(rep(0,N[1]),rep(1,N[2]),rep(2,N[3]))


# Pretreatment of the data
# [sigmamin,sigmamax] this is [0,1] for the logistic function
sigma_min <- 0
sigma_max <- 1
Y_max <- max(Y)
Y_min <- min(Y)

Y_treat <- (sigma_max-sigma_min)/(Y_max-Y_min)*(Y-Y_min)+sigma_min

X_max <- apply(X,2,max)
X_min <- apply(X,2,min)

X_treat <- (sigma_max-sigma_min)/(X_max-X_min)*(X-X_min)+sigma_min
#intento2 <- c()
#for (i in 1:ncol(X)){
#  intento2 <- cbind(intento2, (sigma_max-sigma_min)/(X_max[i]-X_min[i])*(X[,i]-X_min[i])+sigma_min)
#}
#all.equal(X_treat,intento2)