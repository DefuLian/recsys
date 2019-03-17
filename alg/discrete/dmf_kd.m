function [B, D] = dmf_kd(R,varargin)
%dmf_kd discrete matrix factorization with knowledge distillation
%   此处显示详细说明
    [k, max_iter, alpha, ratio, test, top_k] = process_options(varargin, 'K', 64, 'max_iter', 10, ...
    'alpha', 10, 'ratio', 0.1, 'test',[], 'topk', 20);
    [M, N] = size(R);
    [P, Q] = iccf(R, 'alpha', alpha + 1, 'max_iter', max_iter, 'K', k);
    P = P(:, 1:end-2); Q = Q(:,1:end-2);
    %P = 2 *(P>0) - 1; Q = 2 *(Q>0) - 1;
    %P = randn(M, k) * 0.01;
    %Q = randn(N, k) * 0.01;
    if ~isempty(test)
        metric = evaluate_item(R, test, P, Q, 200, 200);
        fprintf('recall@50=%.3f, recall@100=%.3f\n', metric.item_recall(1,50), metric.item_recall(1,100));
    end
    W = +(R>0) * alpha;
    %if ratio > 0
        topk = evaluate_item(R, sparse(M, N), P, Q, top_k, -1);
        R2 = sparse(topk(:,1), topk(:,2), topk(:,3), M, N);
        W = W + R2 * (alpha * ratio);
        R = R + R2;
        %R = +(R>0);
    %end
    R = scale_matrix(R, k);
    [B, D] = kd(R, W, P,  Q, 1, test);
    
end

function R = scale_matrix(R, s)
maxS = max(max(R));
minS = min(R(R~=0));
[I, J, V] = find(R);
if maxS ~= minS
    VV = (V-minS)/(maxS-minS);
    VV = 2 * s * VV - s + 1e-10;
else
    VV = V .* s ./ maxS;
end
R = sparse(I, J, VV, size(R,1), size(R,2));
end

function [B, D]=kd(R, W, B, D, max_iter, test)
Rt = R.';
Wt = W.';

for iter=1:max_iter
    DtD = D.' * D;
    B = optimize_binary(Rt, Wt, D, B, DtD);
    BtB = B.' * B;
    D = optimize_binary(R, W, B, D, BtB);
    loss = loss_();
    fprintf('Iteration=%3d of all optimization, loss=%.1f,', iter, loss);
    if ~isempty(test)
        metric = evaluate_item(R, test, B, D, 200, 200);
        fprintf('recall@50=%.3f, recall@100=%.3f', metric.item_recall(1,50), metric.item_recall(1,100));
    end
    fprintf('\n')
end

function B = optimize_binary(Rt, Wt, D, B, DtD)
m = size(Rt,2);
for u=1:m
    b = B(u,:)';
    r = Rt(:,u);
    w = Wt(:,u);
    idx = w ~= 0;
    w = w(idx);
    r = r(idx);
    if(nnz(idx) == 0)
        Wi = zeros(0);
    else
        Wi = diag(w);
    end
    Du = D(idx, :);
    H = DtD + (Du.' * Wi * Du) ;%+ 0.01 * eye(size(B,2));
    f = Du.' * (r .* w + r);
    B(u,:) = bqp(b, (H+H')/2, f, 'ccd', 1, 2);
    %B(u,:) = H \ f;
end
end

function val = loss_()
    [m, ~] = size(R);
    val = 0;
    for u=1:m
        r = Rt(:,u);
        w = Wt(:,u);
        idx = r ~= 0;
        r_ = D(idx, :) * B(u,:)';
        r = r(idx);
        val = val + sum(((r - r_).^2).* w(idx)); 
    end
    val = val + sum(sum((B'*B) .* (D'*D)));
end

end



