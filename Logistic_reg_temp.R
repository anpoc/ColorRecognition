# The goal of this exercise is creating a color recognition system
# for a data set using a one-vs-all logistic classifier.
# The colour pictures are characterized by a 10 x 10 pixel
# matrix that is stored in the rows. The labels
# corresponding to each digit is 1-blue 2-red 3-green

graphics.off() # close all plots
rm(list=ls()) # remove all objects from from the current workspace (R memory)

# Data

library("rjson")
data <- fromJSON(file = "final_data.json")

## DATA PROCESSING
# Extracting the number of blue, red and green samples.
# N[1]: blues, N[2]: reds, N[3]: greens
N <- unname(sapply(data[[1]],'[[',2))

# Generating the matrix samples X, vector Y 
X <- c()
for (i in 1:length(data[[2]])){
  X <- rbind(X, as.vector(apply(as.matrix(data[[2]][[i]][[4]]), 1, function(x) unlist(x)))) #ASK!!!! t(M) if by row
}

y <- c(rep(1,N[1]),rep(2,N[2]),rep(3,N[3])) #1 - blue, 2 - red, 3 - green



n = dim(X)[1]
q = dim(X)[2]


#for (temp in 1:dim(X)[1]) {
#  X[temp,] = sort(X[temp,])
#}


set.seed(12345)
train_p <- 0.8
TrainingSize <- floor(train_p*nrow(X))
SelectRow <- sample(seq_len(nrow(X)), size = TrainingSize) 

X_train = X[c(SelectRow),]
y_train = y[c(SelectRow)]
X_test = X[-c(SelectRow),]
y_test = y[-c(SelectRow)]


#sorting.. Delete before submitting if not used
#for (temp in 1:dim(X_train)[1]) {
#  X_train[temp,] = sort(X_train[temp,])

#}
#for (temp in 1:dim(X_test)[1]) {
#  X_test[temp,] = sort(X_test[temp,])

#}

n = dim(X_train)[1]
q = dim(X_train)[2]

# maybe it is possible to find the most common HUI value and then decide based on that??

#X <- runif(68500,0,180)
# X = matrix(X,685,100) 
#y <- sample(1:3,685, replace=T) #labels


# One-vs-all logistic regression: logistic classifier for the 3 colours. 


num_labels = 3
beta_one_vs_all = matrix(0,q + 1, num_labels)

for (c in 1:num_labels) {
  
  id_selected=which(y_train==c)
  y_c = y_train
  y_c[-id_selected] = 0  
  y_c[id_selected] = 1
  
  
  
  data = data.frame(y_c,X_train)
  model_glmfit_c = glm(y_c ~., data, start =rep(0,q+1) ,family=binomial(link="logit"),
                       control=list(maxit = 100, trace = FALSE) )
  beta_glmfit_c  = model_glmfit_c$coefficients # NA for linearly dependent vars
  beta_glmfit_c[is.na(beta_glmfit_c)]=0
  
beta_one_vs_all[, c] = beta_glmfit_c 
}

y_classified = apply( cbind(rep(1,n), X_train) %*% beta_one_vs_all , 1, FUN=which.max) # why which.max??? Does not make sense in this case
Empirical_error_one_vs_all = length(which(y_classified != y_train)) / n

n = dim(X_test)[1]
y_classified_test = apply( cbind(rep(1,n), X_test) %*% beta_one_vs_all , 1, FUN=which.max)
Empirical_error_one_vs_all_test = length(which(y_classified_test != y_test)) / n

# Confusion matrix - not working with train test yet.

misclassification_matrix = matrix(0,num_labels, num_labels)
for (i in 1:num_labels) {
for (j in 1:num_labels) {
misclassification_matrix[i, j] = length(which((y == i) & (y_classified == j))) / length(which((y == i)))
}
}

misclassification_matrix_test = matrix(0,num_labels, num_labels)
for (l in 1:num_labels) {
  for (k in 1:num_labels) {
    misclassification_matrix_test[l, k] = length(which((y_test == l) & (y_classified_test == k))) / length(which((y_test == l)))
  }
}



# ETA, MSE, which.max not logical, it classifies the position not the colour, the largest hui is the best
