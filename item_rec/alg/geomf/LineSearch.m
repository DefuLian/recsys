function x = LineSearch(YCY, grad_i, x)
alpha = 1; beta = 0.1;
for iter = 1:10
    grad = grad_i + YCY * x;
    for step =1:20 % search step size
        xn = sparse(max(x - alpha * grad, 0)); d = xn - x;
        gradd = dot(grad, d); dQd = dot(d, YCY * d);
        suff_decr = 0.99 * gradd + 0.5 * dQd < 0;
        if step == 1
            decr_alpha = ~suff_decr; xp = x;
        end
        if decr_alpha
            if suff_decr
                x = xn; break;
            else
                alpha = alpha * beta;
            end
        else
            if ~suff_decr | xp == xn
                x = xp; break;
            else
                alpha = alpha / beta; xp = xn;
            end
        end
    end
end
end
