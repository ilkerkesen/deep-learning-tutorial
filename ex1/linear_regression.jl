module linear_regression
export compute_cost, gradient_descent

function compute_cost(theta, X, y)
    return 0.5 * sum((theta * X - y).^2)
end

function gradient_descent(X, y, alpha, max_iter, min_err)
    n, m = size(X)
    theta = zeros(1, n)
    cost = compute_cost(theta, X, y)

    err, iter = Inf, 0
    while iter <= max_iter && err > min_err
        theta -= alpha * (1/m) * (theta * X - y) * X'
        hold = cost
        cost = compute_cost(theta, X, y)
        err = abs(cost - hold)
        iter += 1
    end
    return theta
end
end
