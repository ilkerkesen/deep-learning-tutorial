# Softmax Regression/Gradient Descent applied to MNIST Dataset

push!(LOAD_PATH, ".")
using MNIST
importall softmax_regression
importall optimization

# parameters
alpha = 0.0000001
max_iter = 10000
min_err = 0.0001

# load MNIST data
println("Loading MNIST dataset...")
X_train, y_train = traindata()
X_test, y_test = testdata()
y_train, y_test = y_train', y_test'

# 0 -> 10, make data more convenient for further operations
y_train[y_train .== 0.0] = 10.0
y_test[y_test .== 0.0] = 10.0
y_train = convert(Array{Int64}, y_train)
y_test = convert(Array{Int64}, y_test)

println("Training...")
tic()
theta = gradient_descent(h, J, X_train, y_train, alpha, max_iter, min_err, 10)
toc()
