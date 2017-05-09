function w = lassononneg(X, y, reg)
grad_i = reg - X.' * y;
L = size(X,2);
w = line_search(X.', sparse(L,L), grad_i, zeros(L,1));
end

function x = line_search(YC, YtY, grad_i, x)
alpha = 1; beta = 0.1;
for iter = 1:5
    grad = grad_i + YC * (x.' * YC).' + YtY * x;
    J = 1:length(grad);
    Ind = grad < 0| x > 0;
    grad = sparse(J(Ind), 1, grad(Ind), length(grad), 1);
    for step =1:10 % search step size
        xn = max(x - alpha * grad, 0); d = xn - x;
        %gradd = dot(grad, d); dQd = dot(d, YtY * d + YC * (d.' * YC).');
        dt = d.';
        gradd = dt * grad;
        dyc = dt * YC; 
        dQd = dt * (YtY * d) + dyc * dyc.';
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
            if ~suff_decr || nnz(xp~=xn)==0
                x = xp; break;
            else
                alpha = alpha / beta; xp = xn;
            end
        end
    end
end
end
