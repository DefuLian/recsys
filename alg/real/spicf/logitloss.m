function v = logitloss(x)
v = log(1+exp(-x));
if nnz(isinf(v))>0
    v(isinf(v)) = -x(isinf(v));
end
end