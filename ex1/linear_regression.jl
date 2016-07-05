module linear_regression
export h, J, init_theta, grad

h(theta, X) = theta' * X
J(theta, X, y) = 0.5 * sum((h(theta, X) - y).^2)
init_theta(n) = rand(n, 1) * 0.001
grad(theta, X, y) = X * (h(theta, X) - y)'

end
end
