module optimization
export gradient_descent

function gradient_descent(h, J, X, y, alpha, max_iter, min_diff, K=1, debug=true)
    n, m = size(X)
    theta = rand(n, K) * 0.001
    cost = J(theta, X, y)

    if K > 1
        idx = map(k -> find(i -> i== k, y), 1:K)
    end

    history = []
    diff, iter = Inf, 1
    cost = J(theta, X, y)
    while iter <= max_iter && diff > min_diff
        if K == 1
            grad = alpha * (1/m) * X * (h(theta, X) - y)'
        else
            hh = h(theta, X)
            grad = zeros(size(theta))
            for k = 1:K
                grad[:, k] = alpha * (1/m) * X[:, idx[k]] * (1 - hh[k, :][idx[k]])
            end
        end

        theta -= grad
        hold = cost
        cost = J(theta, X, y)
        diff = abs(cost - hold)

        if debug
            push!(history, (iter, theta, cost, diff, grad))
            println("Iter #", iter, " ", cost, " ", diff)
        end

        iter += 1
    end
    return (theta, history)
end
end
