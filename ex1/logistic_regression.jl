module logistic_regression
export sigmoid, h, J, init_theta, grad, accuracy

sigmoid(z) = 1 ./ (1 + exp(-z))
h(theta, X) = sigmoid(theta' * X)

function J(theta, X, y)
    hh = h(theta, X)
    return sum(-y .* log(hh) - (1-y) .* log(1-hh))
end

init_theta(n) = rand(n, 1) * 0.001
grad(theta, X, y) = X * (h(theta, X) - y)'
accuracy(theta, X, y) = size(y[:,(h(theta, X) .> 0.5) .== (y .> 0.5)], 2) / size(y, 2)
end
