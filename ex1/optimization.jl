module optimization
export gradient_descent

function gradient_descent(h, J, X, y, alpha, max_iter, min_err)
    n, m = size(X)
    theta = zeros(1, n)
    cost = J(theta, X, y)

    err, iter = Inf, 0
    while iter <= max_iter && err > min_err
        theta -= alpha * (1/m) * (h(theta, X) - y) * X'
        hold = cost
        cost = J(theta, X, y)
        err = abs(cost - hold)
        iter += 1
    end
    return theta
end
end
