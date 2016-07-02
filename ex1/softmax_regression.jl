module softmax_regression
using Debug
export h, J

function h(theta, X)
    mat = exp(theta' * X)
    return 1/sum(mat) .* mat
end

function J(theta, X, y)
    hh = log10(h(theta, X))
    n, m = size(hh)
    I = sub2ind((n,m), collect(y), 1:m)
    return -sum(hh[I])
end
end
