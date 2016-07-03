module softmax_regression
export h, J, predict, accuracy

function h(theta, X)
    mat = exp(theta' * X)
    return (1/sum(mat)) .* mat
end

function J(theta, X, y)
    hh = log10(h(theta, X))
    K, m = size(hh)
    I = sub2ind((K,m), collect(y), 1:m)
    return -sum(hh[I])
end

function predict(theta, X)
    hh = h(theta, X)
    return map(j -> find(i -> i == maximum(hh[:,j]), hh[:,j])[1], 1:size(hh, 2))'
end

function accuracy(y_real, y_pred)
    return sum(map(i -> i == 0 ? 1 : 0, y_real - y_pred)) / size(y_real, 2)
end
end
