function [P, Q, U, V, metric] = iccf(R, varargin)
[M, N] = size(R);
%randn('state', 10);
%rng(10)
init_std = 0.01;
[X, Y, alpha, test, max_iter, K, P, Q, reg_u, reg_i, is_item_fixed, is_user_fixed, ...
    user_bias, item_bias, usr_w, item_w, k_verbose] = ...
   process_options(varargin, 'X', zeros(M,0), 'Y', zeros(N,0), 'alpha', 30, 'test', [], 'max_iter', 20, 'K', 50, 'P', [], 'Q', [], ...
   'reg_u', 0.01, 'reg_i', 0.01, 'is_item_fixed', false, 'is_user_fixed', false, 'user_bias', false, 'item_bias', false,...
   'usr_w', ones(M,1), 'item_w', ones(N,1), 'k-v', 5);

fprintf('iccf(alpha=%d, K=%d, reg_u=%f, reg_i=%f)\n', alpha, K, reg_u, reg_i);

W = GetWeight(R, alpha);
if(isempty(P))
    P = [randn(M, K) * init_std,zeros(M,2)];
end
if(isempty(Q))
    Q = [randn(N, K) * init_std,zeros(N,2)];
end
F = size(X, 2); L = size(Y, 2);
U = [randn(F, K) * init_std, zeros(F,1)]; 
V = [randn(L, K) * init_std, zeros(L,1)]; 

reg_uf = ones(F, 1); 
reg_if = ones(L, 1);

[Iw, Jw, Vw] = find(W);
W = sparse(Iw, Jw, Vw - usr_w(Iw) .* item_w(Jw), M, N);
Wt = W.';
Rt = R.';
if(item_bias)
    bias_M = ones(M,1);
else
    bias_M = zeros(M,1);
end
if(user_bias)
    bias_N = ones(N,1);
else
    bias_N = zeros(N,1);
end
for iter=1:max_iter
    if(~is_user_fixed)
        %P1 = Optimize_CD(Rt, Wt, Q(:,1:K+1), P(:,1:K+1), X, U, reg_u, usr_w, item_w, Q(:,K+2));
        P = Optimize_CD_fast(Rt, Wt, Q(:,1:K+1), P(:,1:K+1), X, U, reg_u, usr_w, item_w, Q(:,K+2));
        U = OptimizeUU(X, P, U, reg_uf);
        P = [P(:,1:(K+1)),bias_M];
    end
    if(~is_item_fixed)
        Q = Optimize_CD_fast(R, W, [P(:,1:K),bias_M], [Q(:,1:K), Q(:,K+2)], Y, V, reg_i, item_w, usr_w, P(:,K+1));
        V = OptimizeUU(Y, Q, V, reg_if);
        Q = [Q(:,1:K), bias_N, Q(:,K+1)];
    end
    fprintf('Iteration=%d of all optimization, loss=%f', iter, fast_loss(R, W, P, Q, usr_w, item_w));
    if mod(iter, k_verbose) ==0 && (~isempty(test))
        metric = evaluate_item(R, test, P, Q, 200, 200);
        if isexplict(test)
            fprintf(',recall@50=%.3f, recall@100=%.3f', metric.item_recall_like(1,50), metric.item_recall_like(1,100));
        else
            fprintf(',recall@50=%.3f, recall@100=%.3f', metric.item_recall(1,50), metric.item_recall(1,100));
        end
    end
    fprintf('\n');
end
if(~isempty(test))
    if mod(max_iter, k_verbose)~=0
        metric = evaluate_item(R, test, P, Q, 200, 200);
    end
end
end

function P = Optimize_CD_fast(R, W, Q, P, X, U, reg_u, a, d, bI)
XU = reg_u * X * U; % M x K
Qs = Q.' * spdiags(d, 0, length(d), length(d)) * Q;
P = iccf_sub(+R, W, Q, P, XU, reg_u, a, d, Qs, bI);
end

function U = OptimizeUU(X, P, Uinit, reg)
F = size(X, 2);
U = Uinit;
if F >0
    if F < 20000 || nnz(X)<0.001 * numel(X)
        t = X.' * P;
        %mat = X.' * X + reg * speye(F, F);
        mat = X.' * X + spdiags(reg,0, F, F);
        U = mat \ t;
    else
        U = ConjGrad(X, P, reg, 'X0', U, 'tol', 1e-6);
    end
end
end

function  W  = GetWeight(R, alpha)
if alpha>200
    W = +(R>0) * alpha;
else
    [M, N] = size(R);
    [I, J, V] = find(R);
    V_t = log10(1+ V * 10.^alpha);
    W = sparse(I, J, V_t, M, N);
end
end

function [X,v] = ConjGrad(A, B, reg, varargin)
% (A' * A + reg * I) * X = A' * B
    [maxit, tol, X] = process_options(varargin, 'maxit', 1e3, 'tol', 1e-3, 'X0', zeros(size(A,2), size(B,2)));
    R = A' * (A * X) + reg * X - A' * B;
    P = -R;
    Rsold = sum(sum(R .^2));
    normb = sqrt(Rsold);
    if normb < eps
        normb = 1;
    end
    v = zeros(maxit,1);
    for i=1:maxit
        Ap = A' * (A * P ) + reg * P;
        alpha = Rsold / sum(sum(P .* Ap));
        X = X + alpha * P;
        R = R + alpha * Ap;
        Rsnew = sum(sum(R .^2));
        normr = sqrt(Rsnew);
        if nargout < 2
            fprintf('iteration %d, norm %f\n', i, normr);
        end
        if normr < normb * tol
              break;
        end
        v(i) = normr;
        P = -R + Rsnew/Rsold * P;
        Rsold = Rsnew;
    end
    v = v(1:i);
end

function loss = Loss(R, W, P, Q, a, d)
[M, N] = size(R);
%[a, d] = process_options(varargin, 'usr_w', ones(M,1) , 'item_w', ones(N,1));
K = size(P,2);
om = ones(M, 1);
on = ones(N, 1);
ok = ones(K, 1);
WR = (W .* R + spdiags(a,0,M, M) * R * spdiags(d,0,N,N));
R2 = R.^2;
WR2 = om.' * ( W .* R2 + spdiags(a,0,M, M) * R2 * spdiags(d,0,N,N) ) * on;
I = on.' * (WR.' * P .* Q) * ok;
L = 0;
Wt = W.';
QtQ = Q.' * spdiags(d, 0, N, N) * Q;
for i=1:M
    w = Wt(:, i);
    p = P(i,:);
    if nnz(w) == 0
        continue;
    end
    Ind = w>0; Wi = diag(w(Ind));    %Wi = repmat(w(Ind), 1, size(V, 2));
    sub_Q = Q(Ind,:);
    QcQ = sub_Q.' * Wi * sub_Q + a(i) * QtQ;
    L = L + p * QcQ * p.';
end
loss = WR2 + L - 2 * I;
end
