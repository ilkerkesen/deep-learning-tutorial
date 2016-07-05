module softmax_regression
export h, J, init_theta, g, predict, accuracy

function h(theta, X)
    mat = exp(theta' * X)
    return (1 ./ sum(mat, 1)) .* mat
end

function J(theta, X, y)
    hh = log10(h(theta, X))
    K, m = size(hh)
    I = sub2ind((K,m), collect(y), 1:m)
    return -sum(hh[I])
end

init_theta(n, K) = rand(n, K) * 0.001

function g(K, y)
    idx = map(k -> find(i -> i == k, y), 1:K)
    function grad(theta, X, y)
        hh = h(theta, X)
        gg = zeros(size(theta))

        for k = 1:K
            gg[:, k] = -X[:, idx[k]] * (1 - hh[k, :][idx[k]])
        end

        return gg
    end

    return grad
end

function predict(theta, X)
    hh = h(theta, X)
    return map(j -> find(i -> i == maximum(hh[:,j]), hh[:,j])[1], 1:size(hh, 2))'
end

accuracy(y_real, y_pred) = sum(map(i -> i == 0 ? 1 : 0, y_real - y_pred)) / size(y_real, 2)
end
