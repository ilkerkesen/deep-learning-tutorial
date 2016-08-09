# Linear Regression applied to Housing Prices Data

push!(LOAD_PATH, ".")
push!(LOAD_PATH, "../common/")

using Gadfly
importall linear_regression
importall optimization

# parameters
alpha = 0.2
max_iter = 100
min_err = 0.0

println("Reading data...")
data = float(open(readdlm, "housing.data"))

data = data'
u = mean(data, 2)
r = maximum(data, 2) - minimum(data, 2)
data = (data .- u) ./ r
n, m = size(data)
data = [ones(1,m); data]
n = n + 1

# shuffle
println("Shuffling data...")
data = data[:, shuffle(collect(1:m))]

# seperate train and test data
m_train = 406
m_test = m - m_train
X_train = data[1:n, 1:m_train]
y_train = data[end,1:m_train]
X_test = data[1:n, m_train+1:end]
y_test = data[end, m_train+1:end]

# theta initialization
theta = init_theta(size(X_train, 1))

# training
println("Training data...")
tic()
theta, history, iter = gradient_descent(h, J, grad, X_train, y_train, theta,
                                        alpha, max_iter, min_err, false)
toc()

ms = collect(1:m_test)
y_pred = theta' * X_test
sorted = sortperm(y_test[:])

println("finished")
println("Epochs: ", iter)
println("Cost (trn): ", J(theta, X_train, y_train))
println("Cost (tst): ", J(theta, X_test, y_test))

# plotting
plot(
    layer(x=ms, y=r[end]*y_test[sorted]+u[end], Geom.point, Theme(default_color=colorant"green")),
    layer(x=ms, y=r[end]*y_pred[sorted]+u[end], Geom.point, Theme(default_color=colorant"red")),
    Guide.XLabel("House"),
    Guide.YLabel("Price"),
    Guide.Title("Housing Prices"),
    Guide.manual_color_key("Prices",["Actual", "Predicted"], ["green", "red"])
)
