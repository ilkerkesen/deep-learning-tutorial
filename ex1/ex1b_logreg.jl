# Logistic Regression with Gradient Descent applied to MNIST Dataset

push!(LOAD_PATH, ".")
push!(LOAD_PATH, "../common")

using MNIST
importall logistic_regression
importall optimization

# parameters
alpha = 0.5
max_iter = 20
min_err = 0.0001

# load MNIST data
println("Loading MNIST dataset...")
X_train, y_train = traindata()
X_test, y_test = testdata()

# filter MNIST data, since it is binary classification
filt_train, filt_test =  y_train .< 2.0, y_test .< 2.0
X_train, y_train = X_train[:,filt_train], y_train[filt_train]
X_test, y_test = X_test[:,filt_test], y_test[filt_test]

# add ones as bias to feature vectors, feature scaling
X_train = [255*ones(1, size(X_train, 2)); X_train] / 255
X_test = [255*ones(1, size(X_test, 2)); X_test] / 255

# transpose results arrays
y_train, y_test = y_train', y_test'

# theta initialization
theta = init_theta(size(X_train, 1))

# training
println("Training...")
tic()
theta, history = gradient_descent(h, J, grad, X_train, y_train, theta,
                                  alpha, max_iter, min_err)
toc()

println("train accuracy: ", accuracy(theta, X_train, y_train))
println("test accuracy: ", accuracy(theta, X_test, y_test))
