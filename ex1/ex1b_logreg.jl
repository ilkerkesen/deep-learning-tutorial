# Logistic Regression with Gradient Descent applied to MNIST Dataset

push!(LOAD_PATH, ".")
using MNIST
importall logistic_regression
importall optimization

# parameters
alpha = 0.000001
max_iter = 2000
min_err = 0.0001

# load MNIST data
println("Loading MNIST dataset...")
X_train, y_train = traindata()
X_test, y_test = testdata()

# filter MNIST data, since it is binary classification
filt_train, filt_test =  y_train .< 2.0, y_test .< 2.0
X_train, y_train = X_train[:,filt_train], y_train[filt_train]
X_test, y_test = X_test[:,filt_test], y_test[filt_test]

# transpose results arrays
y_train, y_test = y_train', y_test'

# training
println("Training...")
tic()
theta = gradient_descent(h, J, X_train, y_train, alpha, max_iter, min_err, 2)
toc()
score = accuracy(theta, X_test, y_test)
println("test accuracy: ", score)
