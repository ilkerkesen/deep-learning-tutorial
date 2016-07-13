using MNIST
sigmoid(z) = 1 ./ (1 + exp(-z))

# load MNIST data
println("Loading MNIST dataset...")
X_train, y_train = traindata()
X_test, y_test = testdata()
y_train, y_test = y_train', y_test'

# mean normalization
X_train /= 255.0
X_test /= 255.0

# make data more convenient for further operations
y_train = y_train .+ 1
y_test = y_test .+ 1
y_train = convert(Array{Int64}, y_train)
y_test = convert(Array{Int64}, y_test)

n, m = size(X_train)
K = 10
h = 256
W1 = 0.01 * rand(h, n)
b1 = zeros(h, 1)
W2 = 0.01 * rand(K, h)
b2 = zeros(K,1)

alpha = 0.2
reg = 0.0

X = X_train
y = y_train

for i = 1:20
    # forward prop 1 hidden layer
    z2 = W1 * X .+ b1 # h x m
    a2 = sigmoid(z2) # h x m
    z3 = W2 * a2 .+ b2 # K x m

    # softmax forw prop
    mat = exp(z3)
    probs = mat ./ sum(mat, 1)

    I = sub2ind((K, m), collect(y), 1:m)
    data_loss = -sum(log(probs[I]))/m
    reg_loss = 0.5 * reg * (sum(W1 .* W1) + sum(W2 .* W2))
    loss = data_loss + reg_loss

    println(loss)

    er3 = probs # K x m
    er3[I] -= 1
    er2 = (W2' * er3) .* (a2 .* (1-a2)) # h x m

    dW2 = er3 * a2' # (K x m) * (m x h) = (K x h)
    db2 = sum(er3, 2)
    dW1 = er2 * X' # (h x m) * (m x n) = (h x n)
    db1 = sum(er2, 2)

    W1 -= alpha * ((dW1 / m) + reg * W1)
    b1 -= alpha * (db1 / m)
    W2 -= alpha * ((dW2 / m) + reg * W2)
    b2 -= alpha * (db2 / m)
end
