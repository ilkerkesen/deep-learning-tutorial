module logistic_regression
export sigmoid, h, J, accuracy

sigmoid(z) = 1 ./ (1 + exp(-z))
h(theta, X) = sigmoid(theta' * X)

function J(theta, X, y)
    hh = h(theta, X)
    return sum(-y .* log(hh) - (1-y) .* log(1-hh))
end

function accuracy(theta, X, y)
    return size(y[:,(h(theta, X) .> 0.5) .== (y .> 0.5)], 2) / size(y, 2)
end
end
