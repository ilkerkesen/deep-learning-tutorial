module linear_regression
export h, J, init_theta, grad

h(theta, X) = theta' * X
J(theta, X, y) = 0.5 * sum((h(theta, X) - y).^2) / size(X, 2)
init_theta(n) = randn(n, 1) * 0.01
grad(theta, X, y) = X * (h(theta, X) - y)' / size(X, 2)

end
