# Linear Regression applied to Housing Prices Data

push!(LOAD_PATH, ".")
push!(LOAD_PATH, "../common/")

using Gadfly
importall linear_regression
importall optimization

# parameters
alpha = 0.0001
max_iter = 100
min_err = 0.0001

println("Reading data...")
data = float(open(readdlm, "housing.data"))

data = data'
n, m = size(data)
data = [ones(1,m); data]
n = n + 1

# shuffle
println("Shuffling data...")
data = data[:, shuffle(collect(1:m))]

# seperate train and test data
m_train = round(Int, floor(m * 0.8))
m_test = m - m_train
X_train = data[1:n, 1:m_train]
y_train = data[end,1:m_train]
X_test = data[1:n, m_train+1:end]
y_test = data[end, m_train+1:end]

# mean normalization
u = mean([X_train X_test], 2)
X_train = X_train ./ u
X_test = X_test ./ u

# theta initialization
theta = init_theta(size(X_train, 1))

# training
println("Training data...")
tic()
theta, history = gradient_descent(h, J, grad, X_train, y_train, theta,
                                  alpha, max_iter, min_err)
toc()

ms = collect(1:m_test)
y_pred = theta' * X_test
sorted = sortperm(y_test[:])

# plotting
plot(
    layer(x=ms, y=y_test[sorted], Geom.point, Theme(default_color=colorant"green")),
    layer(x=ms, y=y_pred[sorted], Geom.point, Theme(default_color=colorant"red")),
    Guide.XLabel("House"),
    Guide.YLabel("Price"),
    Guide.Title("Housing Prices"),
    Guide.manual_color_key("Prices",["Actual", "Predicted"], ["green", "red"])
)
