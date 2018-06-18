function [U, V] = geomf(R, varargin)
[M, N] =size(R);
%randn('state', 10);

[max_iter, alpha, reg_u, reg_i, reg_1, init_std, K, Y] = ...
   process_options(varargin, 'max_iter', 10, 'alpha', 30, 'reg_u', 0.01, ...
                   'reg_i', 0.01, 'reg_1', 0.01, 'init_std', 0.01, 'K', 30, 'Y', zeros(N,0));
Yt = Y.';
U = randn(M, K) * init_std;
V = randn(N, K) * init_std;
W = R * alpha;
Rt = R.';
Wt = W.';
L = size(Yt, 1);
Xt = sparse(L, M);
YtY = Yt * Y;
for iter = 1:1
    [U, V] = optimize_latent(R, W, U, V, Xt, Yt, reg_u, reg_i, max_iter);
    %Xt = optimize_activity(Rt, Wt, Xt, Yt, YtY, zeros(M,0).', zeros(N,0), reg_1);
    Xt = optimize_activity(Rt, Wt, Xt, Yt, YtY, U.', V, reg_1);
    %fprintf('Iteration=%d of all optimization, loss=%f\n', iter, compute_loss(Rt, Wt, U, V, Xt, Y));
end
U = [U, Xt.'];
V = [V, Y];
end

function [U, V] = optimize_latent(R, W, U, V, Xt, Yt, reg_u, reg_i, num_iter)
Wt = W.';
Rt = R.';
K = size(U,2);
for iter = 1:num_iter
    VtV = V.' * V + reg_u * eye(K, K);% NK^2
    YtV = Yt * V; % K|Y|_0
    U = optimize(Rt, Wt, U, V, VtV, YtV, Xt, Yt);
    UtU = U.' * U + reg_i * eye(K, K); % MK^2
    XtU = Xt * U; % K|X|_0
    V = optimize(R, W, V, U, UtU, XtU, Yt, Xt); 
    fprintf('sub-iter %d, loss=%f\n', iter, fast_loss(R, W, U, V));
end

end

function U = optimize(Rt, Wt, U, V, VtV, YtV, Xt, Yt)
% R: NxM sparse matrix, W: NxM matrix, Yt: LxN sparse matrix, Xt: LxM sparse matrix
% U: MxK matrix, V: NxK matrix, VtV: KxK matrix, YtV: LxK matrix
M = size(U,1);
Vt = V.';
XYtV = Xt.' * YtV;
parfor i = 1 : M
    w = Wt(:,i);
    r = Rt(:,i);
    Ind = w>0; 
    if(nnz(Ind) == 0)
        Wi = zeros(0);
    else
        Wi = diag(w(Ind));
    end
    subV = V(Ind,:);
    subYt = Yt(:, Ind);
    subvw = Wi * subV;
    VCV = subV.' * (subvw) + VtV; % N_i K^2
    %YCV = subYt * (Wi * subV) + YtV;
    %Estimate1 = Vt * (w .* r + r) - (Xt(:,i).' * YCV).';
    %Estimate1 = Vt * (w .* r + r) - (Xt(:,i).' * subYt * (subvw) + XYtV(i,:)).';
    Estimate = Vt * (w .* r + r) ; % N_i K
    Estimate = Estimate.' - Xt(:,i).' * subYt * subvw -  XYtV(i,:); %
    u = Estimate / VCV;
    U(i,:) = u;
end

end


function Xt = optimize_activity(Rt, Wt, Xt, Yt, YtY, Ut, V, reg)
YtV = Yt * V;
[L,M] = size(Xt);
user_cell = cell(M,1);
item_cell = cell(M,1);
val_cell = cell(M,1);
for i = 1:M
    w = Wt(:,i);
    r = Rt(:,i);
    Ind = w>0; wi = w(Ind); ri = r(Ind);
    if(nnz(Ind) == 0)
        Wi = zeros(0);
    else
        Wi = spdiags(sqrt(wi), 0, length(wi), length(wi));
    end
    subYt = Yt(:, Ind);
    subV = V(Ind, :);
    YC = subYt * Wi;
    %grad_invariant =  YC * (sqrt(wi) .* (subV * Ut(:,i))) + YtV * Ut(:,i) - subYt * (wi .* ri + ri) + reg;
    %x1 = line_search(YC, YtY, grad_invariant, Xt(:,i));
    grad_invariant =  YC * (sqrt(wi) .* (subV * Ut(:,i))) - subYt * (wi .* ri + ri) + YtV * Ut(:,i)  + reg;
    J = 1:length(grad_invariant);
    ind = grad_invariant<=0;
    grad_invariant = sparse(J(ind), 1, grad_invariant(ind), length(grad_invariant), 1);
    x = line_search(YC, YtY, grad_invariant, Xt(:,i));
    %YCY = YC * YC.' + YtY;
    %YCV = YC * Wi * subV + YtV;
    %grad_invariant =  YCV * Ut(:,i) - Yt * (w .* r + r) + reg;
    %x = LineSearch(YCY, grad_invariant, Xt(:,i));
    [loc, I, val ] = find(x);
    user_cell{i} = i * I;
    item_cell{i} = loc;
    val_cell{i} = val;
end
Xt = sparse(cell2mat(item_cell), cell2mat(user_cell), cell2mat(val_cell), L, M);
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
