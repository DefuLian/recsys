function [P, Q, X, Y, U, V] = qcf(R, varargin)
%UNTITLED2 此处显示有关此函数的摘要
%   此处显示详细说明
[K, max_iter, alpha, test, init_rand] = process_options(varargin, 'K', 64, 'max_iter', 10, ...
    'alpha', 10, 'test',[], 'rand_init',false);
fprintf('qcf(K=%d, max_iter=%d, alpha=%f)\n', K, max_iter, alpha);

rng(200);
C = 256; % fixed number of centers per subspaces (8 bits per subspaces)
D = 8; % each subspace is of d dims
F = K/D; % the number of subspaces
[M, N] = size(R);
%init_rand = false;
if init_rand
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
else
    %[P,Q,X,Y,U,V]=qcf_init_pq(R, 'K', K,'max_iter',max_iter, 'alpha', alpha);
    [P,Q,X,Y,U,V]=qcf_init_opq(R, 'K', K,'max_iter',max_iter, 'alpha', alpha);
end
Rt = R';
W = +(R~=0) * alpha;
Wt = W';
for iter=1:max_iter
    loss = loss_();
    fprintf('Iteration=%3d, loss=%.1f', iter, loss);
    if ~isempty(test)
        if M>20000
            idx = randi(M, 20000, 1);
            metric = evaluate_item(R(idx,:), test(idx,:), P(idx,:), Q, 200, 200);
        else
            metric = evaluate_item(R, test, P, Q, 200, 200);
        end
        if isexplict(test)
            fprintf(', recall@50=%.3f, recall@100=%.3f', metric.item_recall_like(1,50), metric.item_recall_like(1,100));
        else
            fprintf(', recall@50=%.3f, recall@100=%.3f', metric.item_recall(1,50), metric.item_recall(1,100));
        end
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
    [A, f] = get_stat_fast(Rt(:,i), Wt(:,i), St(:,i), P_(:,i), Q, T, QtQ); % O(N_i(D^2 + K))
    % min_c u_c' A u_c - 2 u_c' f, c in {1, 2, ..., C}
    loss = sum((U * A) .* U, 2) - 2 * (U * f); % O(CD^2)
    [~, B_(i)] = min(loss);
end
B = sparse(1:M, B_, true, M, C);
% learn U, the center of user cluster
opts.SYM = true;
for c=1:C
    idx = B(:,c);
    if nnz(idx)>0
        [A, f] = get_stat_fast(Rt(:,idx), Wt(:,idx), St(:,idx), P_(:,idx), Q, T, QtQ);
        %U(c,:) = (A + 1e-3 * eye(D)) \ f; % O(D^3)
        U(c,:) = linsolve(A + 1e-3 * eye(D), f, opts);
    end
end

end

