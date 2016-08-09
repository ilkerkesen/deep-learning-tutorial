module optimization
export gradient_descent

function gradient_descent(h, J, g, X, y, A, alpha, max_iter, min_diff, debug=true)
    n, m = size(X)
    grad = zeros(size(A))
    history = []
    diff, iter = Inf, 0
    cost = J(A, X, y)
    prev = Inf

    while iter < max_iter && diff > min_diff && cost < prev
        grad = g(A, X, y)
        A -= alpha * grad
        prev = cost
        cost = J(A, X, y)
        diff = abs(cost - prev)

        if debug
            push!(history, (iter, A, cost, diff, grad))
            println("Iter #", iter+1, ", J = ", cost)
        end

        iter += 1
    end

    return (A, history, iter)
end
end
