function [B,D]=dmf_admm(R, varargin)
[alpha, max_iter, test, k] = process_options(varargin, 'alpha', 0.01, 'max_iter', 10, 'test',[], 'K', 20);
m = size(R,1);
Rt = R';
B = randn(m,k) * 0.1;
D = [];
debug = true;
for iter=1:max_iter
    D = optimize_binary(R, B, D, alpha);
    B = optimize_real_constraint(Rt, D, alpha);
    loss = loss_(Rt,B,D,alpha);
    if debug
        fprintf('Iteration=%3d of all optimization, loss=%.1f,', iter-1, loss);
        if ~isempty(test)
            %metric = evaluate_rating(test,B,D,10);
            metric = evaluate_item(R, test, +(B>0), D, 200, 200);
            fprintf('recall@100=%.3f, recall@200=%.3f', metric.item_ndcg_like(1,100), metric.item_ndcg_like(1,200));
        end
        fprintf('\n')
    end
end
end

function val = loss_(Rt, B, D, alpha)
    m = size(Rt,2);
    val = 0;
    for u=1:m
        r = Rt(:,u);
        idx = r ~= 0;
        r_ = D(idx, :) * B(u,:)';
        r = r(idx);
        val = val + sum((r - r_).^2) - alpha * sum(r_.^2);
    end
    val = val + alpha*sum(sum((B'*B) .* (D'*D)));
end

function D = optimize_binary(R, B, D, alpha)
n = size(R,2);
k = size(B,2);
BtB = B'*B;
if isempty(D)
    D = +(randn(n,k)>0);
end
for j=1:n
    r = R(:,j); idx = r~=0;
    Bj = B(idx,:);
    H = alpha * BtB + (1 - alpha) * (Bj' * Bj);
    f = Bj' * r(idx);
    %d = bqp([], H, f, 'svr', -1, -1);
    %if isempty(d)
    d = ccd_bqp_mex(D(j,:), H, f, 20);
    %end
    D(j,:) = d;
end
end
function B = optimize_real_constraint(Rt, D, alpha)
rho = 1;
m = size(Rt,2);
k = size(D,2);
Y = zeros(m,k);
B_h = zeros(m,k);
for iter=1:100
    % min_B f(B) + rho |B - (B_h - Y/rho)|_F^2
    B = optimize_real(Rt, D, (B_h-Y/rho)', alpha, rho);
    B_h_old = B_h;
    B_h = B + Y/rho;
    B_h = B_h - mean(B_h);
    %B_h = proj_stiefel_manifold(B_h);
    Y = Y + rho*(B - B_h);
    r_p = norm(B-B_h,'fro');
    r_d = rho*norm(B_h-B_h_old,'fro');
    %ll = loss_(Rt,B,D,alpha) + 2*sum(sum(Y.*(B-B_h))) + rho * norm(B-B_h,'fro')^2;
    if r_p > 5 *r_d
        rho = rho * 2;
    elseif r_d > 5 * r_p
        rho = rho /2;
    end
    if r_p < 0.1 && r_d<0.1
        break
    end
    %fprintf('\tIteration=%3d of admm, loss=%.1f,r_p=%.3f,r_d=%.3f,rho=%f\n', iter, ll, r_p, r_d,rho);
end
end

function B = optimize_real(Rt, D, Xt, alpha, rho)
m = size(Rt,2);
k = size(D, 2);
DtD = D'*D;
B = zeros(m,k);
for i=1:m
    r = Rt(:,i);
    idx = r~=0;
    Di = D(idx,:);
    H = alpha * DtD + (1 - alpha) * (Di' * Di) + rho*eye(k);
    f = Di'*r(idx) + rho * Xt(:,i);
    B(i,:) = H \ f;
end
end