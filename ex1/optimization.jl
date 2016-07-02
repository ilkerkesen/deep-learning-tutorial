module optimization
export gradient_descent

function gradient_descent(h, J, X, y, alpha, max_iter, min_err, K=1)
    n, m = size(X)
    theta = zeros(n, K)
    cost = J(theta, X, y)

    if K > 1
        idx = map(k -> find(i -> i== k, y), 1:K)
    end

    err, iter = Inf, 0
    while iter <= max_iter && err > min_err
        if K == 1
            theta -= alpha * (1/m) * X * (h(theta, X) - y)'
        else
            hh = h(theta, X)
            for k = 1:K
                theta[:,k] -= alpha * (1/m) * X[:,idx[k]] * (1 - hh[K,:][idx[k]])
            end
        end
        hold = cost
        cost = J(theta, X, y)
        err = abs(cost - hold)
        iter += 1
        println(iter)
    end
    return theta
end
end
