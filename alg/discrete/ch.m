function [B,D]= ch(R,varargin)
[max_iter, k, debug] = process_options(varargin, 'max_iter', 20, 'K', 20, 'debug', true);
print_info();
[m,n] = size(R);
if max(R(R~=0)) > min(R(R~=0)) + 1e-3
    [I,J,V]=find(R);
    V = (V - min(V)) ./ (max(V) - min(V));
    R = sparse(I,J,V,m,n);
end
converge = false;
B = randn(m, k)*0.1; D = randn(n, k)*0.1;
it = 1;
loss0 = inf;
while ~converge
    B = sqrt(m)*proj_stiefel_manifold(R*D);
    D = sqrt(n)*proj_stiefel_manifold(R'*B);
    loss = loss_();
    if debug
        fprintf('Iteration=%3d of all optimization, loss=%.1f\n', it, loss);
    end
    if it >= max_iter || abs(loss0-loss)<1
        converge = true;
    end
    it = it + 1;
    loss0 = loss;
end
[B,D] = rounding(B,D);
function print_info()
    fprintf('ch (K=%d, max_iter=%d)\n', k, max_iter);
end
function v = loss_()
    v = sum(sum(R.*R)) - 2* sum(sum(B .* (R*D))) + m*n*k;
end
end
function [B,D] = rounding(B,D)
B = 2*(B>0)-1;
D = 2*(D>0)-1;
end