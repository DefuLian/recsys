function [P, Q, X, Y, D, V] = qcf1(R, varargin)
%UNTITLED2 此处显示有关此函数的摘要
%   此处显示详细说明
[K, max_iter, alpha, test] = process_options(varargin, 'K', 64, 'max_iter', 10, ...
    'alpha', 10, 'test',[]);
rng(200);
C = 256; % fixed number of centers per subspaces (8 bits per subspaces)
D = 8; % each subspace is of d dims
F = K/D; % the number of subspaces
[M, N] = size(R);
X = randi(C, M, F);
Y = randi(C, N, F);
U = randn(C, F*D) * 0.01;
V = randn(C, F*D) * 0.01;
P = zeros(M, K);
Q = zeros(N, K);
for f=1:F
    idx = (f-1)*D+(1:D);
    P(:, idx) = U(X(:,f), idx);
    Q(:, idx) = V(Y(:,f), idx);
end
Rt = R';
W = +(R~=0) * alpha;
Wt = W';
for iter=1:max_iter
    loss = loss_();
    fprintf('Iteration=%3d of all optimization, loss=%.1f', iter, loss);
    if ~isempty(test)
        metric = evaluate_item(R, test, P, Q, 200, 200);
        fprintf('recall@50=%.3f, recall@100=%.3f', metric.item_recall(1,50), metric.item_recall(1,100));
    end
    fprintf('\n')
    for f=1:F
        S = P(:,[1:(f-1)*D, (f*D+1):F*D]);
        T = Q(:,[1:(f-1)*D, (f*D+1):F*D]);
        idx = (f-1)*D+(1:D);
        Q_ = Q(:,idx); U_ = U(:,idx);
        [X_, U_] = qmf_(Rt, Wt, U_, Q_, S', T);
        X(:,f) = X_; U(:,idx) = U_;
        P_ = U_(X_,:);
        P(:,idx) = P_;
        V_ = V(:,idx);
        [Y_, V_] = qmf_(R, W, V_, P_, T', S);
        Y(:,f) = Y_; V(:,idx) = V_;
        Q(:,idx) = V_(Y_,:);
    end
end

function val = loss_()
    val = 0;
    for u=1:size(R, 1)
        r = Rt(:,u);
        w = Wt(:,u);
        idx_r = r ~= 0;
        r_ = Q(idx_r, :) * P(u,:)';
        r = r(idx_r);
        val = val + sum(((r - r_).^2).* w(idx_r)); 
    end
    val = val + sum(sum((P'*P) .* (Q'*Q)));
end

end

function [B_, U] = qmf_(Rt, Wt, U, Q, St, T)
% B:MxC, U: CxD, Q: NxD, S: Mx(K-D), T: Nx(K-D)
% learn B, the assignment of users to cluster
M = size(Rt, 2);
QtQ = Q' * Q; % DxD, O(ND^2)
P_ = Q' * T * St; % DxM, O((M+N)DK)
[C,D] = size(U);
B_ = zeros(M, 1);
for i=1:M
    %[A, f] = get_stat_fast(Rt(:,i), Wt(:,i), St(:,i), P_(:,i), Q, T, QtQ);
    [A, f] = get_stat(Rt(:,i), Wt(:,i), St(:,i), P_(:,i), Q, T, QtQ); % O(N_i(D^2 + K))
    % min_c u_c' A u_c - 2 u_c' f, c in {1, 2, ..., C}
    loss = sum((U * A) .* U, 2) - 2 * (U * f); % O(CD^2)
    [~, B_(i)] = min(loss);
end
B = sparse(1:M, B_, 1, M, C);
% learn U, the center of user cluster

for c=1:C
    users = find(B(:,c))';
    if ~isempty(users)
        A = 1e-3 * eye(D);
        f = zeros(D, 1);
        for i = users
            [A_, f_] = get_stat(Rt(:,i), Wt(:,i), St(:,i), P_(:,i), Q, T, QtQ);
            A = A + A_;
            f = f + f_;
        end
        U(c,:) = A \ f; % O(D^3)
    end
end

end

function [A,b] = get_stat(r, w, s, p, Q, T, QtQ)
    idx = w ~= 0;
    w = w(idx);
    r = r(idx);
    if(nnz(idx) == 0)
        Wi = zeros(0);
    else
        Wi = diag(w);
    end
    Qi = Q(idx,:);
    Ti = T(idx,:);
    A = QtQ + Qi' * Wi * Qi; % N_i D^2
    %b = Qi' * (r .* w + r) - Qi' * ((Ti * s') .* w) - p; % N_i K
    b = Qi' * ((r - Ti * s) .* w + r) - p; % N_i K
end
