# Softmax Regression/Gradient Descent applied to MNIST Dataset

push!(LOAD_PATH, ".")
using MNIST
importall softmax_regression
importall optimization

# parameters
alpha = 0.1
max_iter = 500
min_err = 0.000001

# load MNIST data
println("Loading MNIST dataset...")
X_train, y_train = traindata()
X_test, y_test = testdata()
y_train, y_test = y_train', y_test'

# bias feature, mean normalization
X_train = [255*ones(1, size(X_train, 2)); X_train] / 255.0
X_test = [255*ones(1, size(X_test, 2)); X_test] / 255.0

# make data more convenient for further operations
y_train = y_train .+ 1
y_test = y_test .+ 1
y_train = convert(Array{Int64}, y_train)
y_test = convert(Array{Int64}, y_test)

println("Training...")
tic()
theta, history = gradient_descent(h, J, X_train, y_train, alpha, max_iter, min_err, 10)
toc()
