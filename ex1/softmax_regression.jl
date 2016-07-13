module softmax_regression
export h, J, init_theta, g, predict, accuracy

function h(theta, X)
    mat = exp(theta' * X)
    return (1 ./ sum(mat, 1)) .* mat
end

function J(theta, X, y)
    hh = h(theta, X)
    K, m = size(hh)
    I = sub2ind((K,m), collect(y), 1:m)
    return -sum(log(hh[I]))/m
end

init_theta(n, K) = randn(n, K) * 0.01

function g(K, y)
    idx = map(k -> find(i -> i == k, y), 1:K)
    function grad(theta, X, y)
        hh = h(theta, X)
        K, m = size(hh)
        I = sub2ind((K,m), collect(y), 1:m)
        hh[I] -= 1
        return (X * hh') / m
    end

    return grad
end

function predict(theta, X)
    hh = h(theta, X)
    return map(j -> find(i -> i == maximum(hh[:,j]), hh[:,j])[1], 1:size(hh, 2))'
end

accuracy(y_real, y_pred) = sum(map(i -> i == 0 ? 1 : 0, y_real - y_pred)) / size(y_real, 2)
end
