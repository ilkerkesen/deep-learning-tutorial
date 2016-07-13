# Softmax Regression/Gradient Descent applied to MNIST Dataset

push!(LOAD_PATH, ".")
push!(LOAD_PATH, "../common")

using MNIST
importall softmax_regression
importall optimization

# parameters
alpha = 0.2
max_iter = 200
min_err = 0.0

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

# theta initialization
theta = init_theta(size(X_train, 1), 10)

println("Training...")
tic()
theta, history = gradient_descent(h, J, g(10, y_train), X_train, y_train,
                                  theta, alpha, max_iter, min_err)
toc()

println("train accuracy: ", accuracy(predict(theta, X_train), y_train))
println("test accuracy: ", accuracy(predict(theta, X_test), y_test))
