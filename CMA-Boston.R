# Library preprocessing and seed configuration

library(tidyverse)
library(MASS)
library(class)
library(randomForest)
library(tree)
library(scales)
set.seed(100383186)
# EXERCISE 1 -- 0.5/10 --#

# Load Boston dataframe
df_raw <- Boston

# Extract crim median from the dataset
crim_raw_median <- median(df_raw$crim)

# Create both prob distributions with 70% chance for 1 and for 0
prob_distr_crim_below <- sample(
    c(0, 1),
    size = nrow(df_raw), replace = TRUE, prob = c(0.7, 0.3)
)

prob_distr_crim_above <- sample(
    c(1, 0),
    size = nrow(df_raw), replace = TRUE, prob = c(0.7, 0.3)
)

# If above the median n or below median, respectively (using prob distributions)
df_raw$crim <- ifelse(df_raw$crim > crim_raw_median,
    prob_distr_crim_above, prob_distr_crim_below
)

# EXERCISE 2 -- 1/10 --#

# Distribution for row selection -- Will take row with a prob of 80%
# row_sel <- sample(c(TRUE, FALSE),
#    size = nrow(df_raw), prob = c(0.8, 0.2),
#    replace = TRUE
# )
# Assign the randomly selected rows and divide
# df_train <- df_raw[row_sel, ]
# df_test <- df_raw[!row_sel, ]
# Now, repeat the prob sample until it gives approx 80%, although
# an approximate value is also accepted.

# However I prefere to proceed as follows:
# Take a random vector sample with numbers from 1 to the number of obs.
samp <- unique(sample(nrow(df_raw), size = nrow(df_raw)))
# Used unique to avoid repetition

# The 80% first values are assigned to train, and the other 20% to test
df_train <- df_raw[samp[1:(506 * 0.8)], ]
df_test <- df_raw[samp[(506 * 0.8):506], ]


# EXERCISE 3 -- 2/10 --#
summary(df_train)

lrm <- glm(crim ~ zn + indus + chas + nox + rm + age + dis + rad + tax + ptratio + black + lstat + medv,
    data = df_train,
    family = binomial
)
summary(lrm)

pred_raw_lr <- predict(lrm, newdata = df_test, type = "response")

cutoff <- 0.6
pred_lr <- ifelse(pred_raw_lr > cutoff, 1, 0)


conf_matrix_lr <- table(df_test$crim, pred_lr)

accuracy_lr <- ((conf_matrix_lr[2, 2] + conf_matrix_lr[1, 1])
/ (nrow(df_test)))

sensitivity_lr <- (conf_matrix_lr[2, 2]
/ (conf_matrix_lr[2, 2] + conf_matrix_lr[1, 2]))

specificity_lr <- (conf_matrix_lr[1, 1]
/ (conf_matrix_lr[1, 1] + conf_matrix_lr[2, 1]))

# EXERCISE 4 -- 2.5/10 --#
lrm4 <- glm(crim ~ age,
    data = df_train,
    family = binomial
)
summary(lrm4)
pred_raw_lr4 <- predict(lrm4, newdata = df_test, type = "response")
pred_lr4 <- ifelse(pred_raw_lr4 > cutoff, 1, 0)
conf_matrix_lr4 <- table(df_test$crim, pred_lr4)
accuracy_lr4 <- ((conf_matrix_lr4[2, 2] + conf_matrix_lr4[1, 1])
/ (nrow(df_test)))


# EXERCISE 5 -- 3.5/10 --#
ldam <- lda(crim ~ zn + indus + chas + nox + rm + age + dis + rad + tax + ptratio + black + lstat + medv,
    data = df_train
)
summary(ldam)

pred_lda <- predict(ldam, newdata = df_test)

conf_matrix_lda <- table(df_test$crim, pred_lda$class)

accuracy_lda <- ((conf_matrix_lda[2, 2] + conf_matrix_lda[1, 1])
/ (nrow(df_test)))

sensitivity_lda <- (conf_matrix_lda[2, 2]
/ (conf_matrix_lda[2, 2] + conf_matrix_lda[1, 2]))

specificity_lda <- (conf_matrix_lda[1, 1]
/ (conf_matrix_lda[1, 1] + conf_matrix_lda[2, 1]))

# EXERCISE 6 -- 5/10 --#
K <- 20

conf_matrix_knn <- vector(mode = "list", length = K)
accuracy_knn <- rep(NA, length = K)
sensitivity_knn <- rep(NA, length = K)
specificity_knn <- rep(NA, length = K)

for (i in 1:K) {
    knnm <- knn(train = df_train, test = df_test, cl = df_train$crim, k = i)

    conf_matrix_knn[[i]] <- table(df_test$crim, knnm)

    accuracy_knn[i] <- ((conf_matrix_knn[[i]][2, 2] +
        conf_matrix_knn[[i]][1, 1]) / (nrow(df_test)))

    sensitivity_knn[i] <- (conf_matrix_knn[[i]][2, 2]
    / (conf_matrix_knn[[i]][2, 2] + conf_matrix_knn[[i]][1, 2]))

    specificity_knn[i] <- (conf_matrix_knn[[i]][1, 1]
    / (conf_matrix_knn[[i]][1, 1] + conf_matrix_knn[[i]][2, 1]))
}

# EXERCISE 7 -- 5.5/10 --#

# Group into a data frame in order to properly visualize using ggplot
df_knn <- data.frame(
    k = seq(1, 20),
    accuracy = accuracy_knn,
    sensitivity = sensitivity_knn,
    specificity = specificity_knn
)

# Plot of accuracy
ggplot(data = df_knn) +
    geom_line(aes(y = accuracy, x = k),
        col = "firebrick3",
        linewidth = 1
    ) +
    scale_x_continuous(breaks = (c(seq(0, 21)))) +
    labs(
        title = "Accuracy measurements for each K in KNN model",
        y = "Accuracy", x = "K"
    ) +
    theme(
        plot.title = element_text(size = 24, hjust = 0.5, family = "Times New Roman"),
        axis.title.x = element_text(size = 20, family = "Times New Roman", face = "bold"),
        axis.title.y = element_text(size = 20, family = "Times New Roman", face = "bold"),
        axis.text = element_text(size = 16),
    )
# Plot of sensitivity
ggplot(data = df_knn) +
    geom_line(aes(y = sensitivity, x = k),
        col = "firebrick3",
        linewidth = 1
    ) +
    scale_x_continuous(breaks = (c(seq(0, 21)))) +
    labs(
        title = "Sensitivity measures for each K in KNN model",
        y = "Sensitivity",
        x = "K"
    ) +
    theme(
        plot.title = element_text(size = 24, hjust = 0.5, family = "Times New Roman"),
        axis.title.x = element_text(size = 20, family = "Times New Roman", face = "bold"),
        axis.title.y = element_text(size = 20, family = "Times New Roman", face = "bold"),
        axis.text = element_text(size = 16)
    )

# Plot of specificity
ggplot(data = df_knn) +
    geom_line(aes(y = specificity, x = k),
        col = "firebrick3",
        linewidth = 1
    ) +
    scale_x_continuous(breaks = (c(seq(0, 21)))) +
    labs(
        title = "Specificity measures for each K in KNN model",
        y = "Specificity",
        x = "K"
    ) +
    theme(
        plot.title = element_text(size = 24, hjust = 0.5, family = "Times New Roman"),
        axis.title.x = element_text(size = 20, family = "Times New Roman", face = "bold"),
        axis.title.y = element_text(size = 20, family = "Times New Roman", face = "bold"),
        axis.text = element_text(size = 16)
    )
# They will be used later for identification of best K value

# Exercise 8 -- 6.5/10 --#
forestm <- randomForest(as.factor(crim) ~ .,
    data = df_train,
    ntree = 100,
    mtry = 5,
    importance = TRUE
)

pred_rf <- predict(forestm, newdata = df_test)

conf_matrix_rf <- table(df_test$crim, pred_rf)

accuracy_rf <- ((conf_matrix_rf[2, 2] + conf_matrix_rf[1, 1])
/ (nrow(df_test)))

sensitivity_rf <- (conf_matrix_rf[2, 2]
/ (conf_matrix_rf[2, 2] + conf_matrix_rf[1, 2]))

specificity_rf <- (conf_matrix_rf[1, 1]
/ (conf_matrix_rf[1, 1] + conf_matrix_rf[2, 1]))

# Exercise 9 -- 9/10 --#

# Relevance of predictors
imp_var <- data.frame(importance(forestm))
imp_var <- imp_var[order(-imp_var$MeanDecreaseAccuracy), ]
varImpPlot(forestm)
imp_name <- c("crim", rownames(imp_var))

# Establish data frames with the order of variables as specified
## ---  REMEMBER TO CHECK IF NEEDED FOR OUTPUT VARIABLE CHANGE --#

ordered_df_train <- data.frame(matrix(nrow = nrow(df_train), ncol = ncol(df_train)))
ordered_df_test <- data.frame(matrix(nrow = nrow(df_test), ncol = ncol(df_test)))
ordered_treedf_train <- data.frame(matrix(nrow = nrow(df_train), ncol = ncol(df_train)))
ordered_treedf_test <- data.frame(matrix(nrow = nrow(df_test), ncol = ncol(df_test)))

for (t in 1:14) {
    colnames(ordered_df_train) <- imp_name
    colnames(ordered_df_test) <- imp_name
    colnames(ordered_treedf_train) <- imp_name
    colnames(ordered_treedf_test) <- imp_name

    if (t == 1) {
        ordered_df_train[, t] <- df_train$crim
        ordered_df_test[, t] <- df_test$crim
        ordered_treedf_test[, t] <- as.factor(df_test$crim)
        ordered_treedf_train[, t] <- as.factor(df_train$crim)
    } else {
        ordered_df_train[, t] <- df_train[imp_name[t]]
        ordered_df_test[, t] <- df_test[imp_name[t]]
        ordered_treedf_train[, t] <- df_train[imp_name[t]]
        ordered_treedf_test[, t] <- df_test[imp_name[t]]
    }
}

order <- colnames(ordered_df_train)[2:14]

# Then create the lists of the confusion matrices
list_lda <- vector(mode = "list", length = 13)
list_lr <- vector(mode = "list", length = 13)
list_knn <- vector(mode = "list", length = 13)
list_tree <- vector(mode = "list", length = 13)

# Predictions for lda and lr
pred_vect_lr <- vector(mode = "list", length = 13)
pred_vect_lda <- vector(mode = "list", length = 13)
pred_vect_tree <- vector(mode = "list", length = 13)

# Create the empty vectors for accuracy values
accuracyvector_lda <- rep(NA, length = 13)
accuracyvector_lr <- rep(NA, length = 13)
accuracyvector_knn <- rep(NA, length = 13)
accuracyvector_tree <- rep(NA, length = 13)

# Formula and model declarations
list_formula <- vector(mode = "list", length = 13)
glm_list <- vector(mode = "list", length = 13)
ldam_list <- vector(mode = "list", length = 13)
knnm_list <- vector(mode = "list", length = 13)
treeclassm_list <- vector(mode = "list", length = 13)


for (j in 1:13) {
    list_formula[[j]] <- reformulate(order[1:j], response = "crim")

    glm_list[[j]] <- glm(formula = list_formula[[j]], data = ordered_df_train, family = binomial)
    pred_vect_lr[[j]] <- ifelse(predict(glm_list[[j]], newdata = ordered_df_test) > cutoff, 1, 0)
    list_lr[[j]] <- table(ordered_df_test$crim, pred_vect_lr[[j]])
    accuracyvector_lr[[j]] <- ((list_lr[[j]][2, 2] + list_lr[[j]][1, 1])
    / (nrow(ordered_df_test)))

    ldam_list[[j]] <- lda(formula = list_formula[[j]], data = ordered_df_train, family = binomial)
    pred_vect_lda[[j]] <- predict(ldam_list[[j]], newdata = ordered_df_test)
    list_lda[[j]] <- table(ordered_df_test$crim, pred_vect_lda[[j]]$class)
    accuracyvector_lda[[j]] <- ((list_lda[[j]][2, 2] + list_lda[[j]][1, 1])
    / (nrow(ordered_df_test)))

    knnm_list[[j]] <- knn(train = ordered_df_train[, c(1:(j + 1))], test = ordered_df_test[, c(1:(j + 1))], cl = ordered_df_train$crim, k = 3)
    list_knn[[j]] <- table(ordered_df_test$crim, knnm_list[[j]])
    accuracyvector_knn[[j]] <- ((list_knn[[j]][2, 2] + list_knn[[j]][1, 1])
    / (nrow(ordered_df_test)))

    treeclassm_list[[j]] <- tree(list_formula[[j]], data = ordered_treedf_train)
    pred_vect_tree[[j]] <- predict(treeclassm_list[[j]], newdata = ordered_treedf_test, type = "class")
    list_tree[[j]] <- table(ordered_df_test$crim, pred_vect_tree[[j]])
    accuracyvector_tree[[j]] <- ((list_tree[[j]][2, 2] + list_tree[[j]][1, 1])
    / (nrow(ordered_df_test)))
}

# Exercise 10 --- 9.5/10 --#

# Now we need to plot the results accordingly to decide which is the best in terms of accuracy
df_models <- data.frame(
    experiment = seq(1, 13, 1),
    accuracy_lr = accuracyvector_lr,
    accuracy_knn = accuracyvector_knn,
    accuracy_lda = accuracyvector_lda,
    accuracy_tree = accuracyvector_tree
)


# Plot of accuracy for logical regresion models
ggplot(data = df_models) +
    geom_line(aes(y = accuracy_lr, color = "Logical Regression", x = experiment),
        linewidth = 1,
        show.legend = TRUE
    ) +
    scale_x_continuous(breaks = (c(seq(0, 13)))) +
    ylim(0.5, 1) +
    labs(
        title = "Accuracy measurements for each model",
        y = "Accuracy", x = "Experiment"
    ) +
    theme(
        plot.title = element_text(size = 24, hjust = 0.5, family = "Times New Roman"),
        axis.title.x = element_text(size = 20, family = "Times New Roman", face = "bold"),
        axis.title.y = element_text(size = 20, family = "Times New Roman", face = "bold"),
        axis.text = element_text(size = 16),
        legend.title = element_text(size = 20, hjust = 0.5, family = "Times New Roman", face = "bold"),
        legend.text = element_text(size = 16, family = "Times New Roman")
    ) +
    geom_line(aes(y = accuracy_lda, color = "Linear Discriminant Analysis", x = experiment),
        show.legend = TRUE,
        linewidth = 1
    ) +
    geom_line(aes(y = accuracy_knn, color = "K-Nearest Neighbours", x = experiment),
        show.legend = TRUE,
        linewidth = 1
    ) +
    geom_line(aes(y = accuracy_tree, color = "Classification Tree", x = experiment),
        show.legend = TRUE,
        linewidth = 1
    ) +
    scale_color_discrete("Model used")
# Exercise 11 --10/10 -- #
