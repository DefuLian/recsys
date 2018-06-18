function [ U, V] = graph_wals(train, varargin )

%randn('state', 10);
R = train>0;
[M, N] = size(R);
[max_iter, reg_u, reg_i, Su,  Si, alpha, eta_u, eta_i, K, init_std] = process_options(varargin, 'max_iter', 10, 'reg_u', 0.01, 'reg_i', 0.01, ...
    'user_sim', sparse(M,M), 'item_sim', sparse(N,N), 'alpha', 10, 'eta_u', 0.01, 'eta_i', 0.01, 'K', 50,'init_std', 0.01);

W = R * alpha;
Wt = W.';
Rt = R.';
U = randn(M, K) * init_std;
V = randn(N, K) * init_std;

for iter = 1:max_iter
    UtU = U.' * U + reg_i * eye(K);
    V = Optimize(R, W, V, U, UtU, eta_i * Si);
    VtV = V.' * V + reg_u * eye(K);
    U = Optimize(Rt, Wt, U, V, VtV, eta_u * Su);
    
    fprintf('Iteration=%d, loss=%f\n', iter, fast_loss(R, W, U, V));
end
end
function U = Optimize(R, W, U, V, VtV, Su)
su = sum(Su);
[~, M] = size(W);
Vt = V.';
K = size(U,2);
for i = randperm(M)
    w = W(:, i);
    r = R(:, i);
    s = Su(:,i);
    Ind = w>0; 
    if nnz(w) == 0
        Wi = zeros(0);
    else
        Wi = diag(w(Ind));
    end
    sub_V = V(Ind,:);
    VCV = sub_V.' * Wi * sub_V + VtV + su(i) * eye(K); 
    Y = Vt * (w .* r + r) + U.' * s;
    U(i,:) = VCV \ Y;
end
end


