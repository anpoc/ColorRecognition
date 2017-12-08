# The goal of this exercise is creating a color recognition system
# for a data set using a one-vs-all logistic classifier.
# The colour pictures are characterized by a 10 x 10 pixel
# matrix that is stored in the rows. The labels
# corresponding to each digit is 1-blue 2-red 3-green

graphics.off() # close all plots
rm(list=ls()) # remove all objects from from the current workspace (R memory)

# Data

X <- runif(68500,0,180)
X = matrix(X,685,100) 
y <- sample(1:3,685, replace=T) #labels

n = dim(X)[1]
q = dim(X)[2]

# 4.- One-vs-all logistic regression: logistic classifier for the 3 colours. 


num_labels = 3
beta_one_vs_all = matrix(0,q + 1, num_labels)

for (c in 1:num_labels) {
  
  id_selected=which(y==c)
  y_c = y
  y_c[-id_selected] = 0  
  y_c[id_selected] = 1
  
  data = data.frame(y_c,X)
  model_glmfit_c = glm(y_c ~., data, start =rep(0,q+1) ,family=binomial(link="logit"),
                       control=list(maxit = 100, trace = FALSE) )
  beta_glmfit_c  = model_glmfit_c$coefficients # NA for linearly dependent vars
  beta_glmfit_c[is.na(beta_glmfit_c)]=0
  
beta_one_vs_all[, c] = beta_glmfit_c 
}

y_classified = apply( cbind(rep(1,n), X) %*% beta_one_vs_all , 1, FUN=which.max)
Empirical_error_one_vs_all = length(which(y_classified != y)) / n


# Confusion matrix

misclassification_matrix = matrix(0,num_labels, num_labels)
for (i in 1:num_labels) {
for (j in 1:num_labels) {
misclassification_matrix[i, j] = length(which((y == i) & (y_classified == j))) / length(which((y == i)))
}
}



