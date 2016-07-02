module linear_regression
export h, J

h(theta, X) = theta' * X
J(theta, X, y) = 0.5 * sum((h(theta, X) - y).^2)
end
