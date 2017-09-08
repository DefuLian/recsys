function [P, Q] = spicf(R, varargin)
[alpha, test, max_iter, K, reg_u, reg_i, is_hard] = process_options(varargin, 'alpha', 50, 'test', [], 'max_iter', 20, 'K', 50, ...
   'reg_u', 0.01, 'reg_i', 0.01, 'is_hard', true);
mu = 0.1;
[M, N] = size(R);
a = ones(M,1); d = ones(N,1);
[P, Q] = piccf(R, 'alpha', alpha, 'max_iter', 10, 'K', K, 'reg_u', reg_u, 'reg_i', reg_i, 'usr_w', a, 'item_w', d);
inner_max_iter = 10;
Rt = R.';
for i=1:max_iter
    %P = randn(M, K) * 0.01; Q = randn(N, K) * 0.01;
    eval = evaluate_item(R, test, P, Q, 200, 200);
    fprintf('ndcg@50=%f\n', eval.ndcg(50))
    pace = initial_pace(R, P, Q, a, d);
    iter = 1;
    d = ones(N,1);
    while(true)
        a = optimize_theta(Rt,  P,  Q,  d, pace, is_hard);
        fprintf('Iteration=%d of spl for theta, pace=%f, |theta|=%d\n', iter, pace, sum(a>0))
        [P, Q] = piccf(R, 'alpha', alpha, 'max_iter', inner_max_iter, 'K', K, 'reg_u', reg_u, 'reg_i', reg_i,...
            'P', P, 'Q', Q, 'usr_w', a, 'item_w', d);
        pace = pace * mu;
        iter = iter + 1;
        if nnz(a) == M
            break;
        end
    end
    pace = initial_pace(R, P, Q, a, d);
    iter = 1;
    a = ones(M, 1);
    while(true)
        d = optimize_theta(R,  Q,  P,  a, pace, is_hard);
        fprintf('Iteration=%d of spl for beta, pace=%f, |beta|=%d\n', iter, pace, sum(d>0))
        [P, Q] = piccf(R, 'alpha', alpha, 'max_iter', inner_max_iter, 'K', K, 'reg_u', reg_u, 'reg_i', reg_i,...
            'P', P, 'Q', Q, 'usr_w', a, 'item_w', d);
        pace = pace * mu;
        iter = iter + 1;
        if nnz(d) == N
            break;
        end
    end
    
    
end
end

function [a, d] = optimize_theta_beta(R, P, Q, a, d, pace, is_hard)
Rt = R.';
prev_loss = 0;
for iter = 1:10
    l = sploss(R, P, Q, a, d, pace, is_hard);
    fprintf('Iteration=%d of self-paced learning, loss=%f, |theta|=%d, |beta|=%d\n', iter, ...
        l, sum(a>0), sum(d>0))
    if abs(prev_loss-l)< eps
        break
    end
    prev_loss = l;
end
end
function a = optimize_theta(Rt, P, Q, d, pace, is_hard)
[N, M] = size(Rt);
a = zeros(M, 1);
[J, I, ~] = find(Rt);
pred = sparse(J, I, sum(P(I,:).* Q(J,:), 2), N, M); 
loss = pred.^2;
Qt = Q.';
Qs = Qt * spdiags(d, 0, N, N) * Q;
d_sum = sum(d);
d2_sum = sum(d.^2);
parfor i=1:M
    r = Rt(:, i);
    ind = r>0;
    p = P(i,:);
    v = loss(:, i);
    sub_d = d(ind);
    sub_r = r(ind);
    sub_rd = sub_r.*sub_d;
    avg_loss = sum(sub_r .* sub_rd);
    avg_loss = avg_loss -  p * ( Qt(:,ind) * sub_rd) * 2;
    avg_loss = avg_loss + sum( (p * Qs) .* p);
    avg_loss = avg_loss - sum(v(ind) .* d(ind));
    d_neg_sum = d_sum - sum(d(ind));
    avg_loss = avg_loss / d_neg_sum;
    if avg_loss < 1/pace
        if is_hard
            a(i) = 1;
        else
            pace_large = pace * d_neg_sum / (d_neg_sum - d2_sum + sum(ind .* d .* d) ) ;
            if avg_loss < 1/pace_large
                a(i) = 1;
            else
                a(i) = pace_large / (pace_large-pace) * (1 - pace * avg_loss);
            end
        end
    else
        a(i) = 0;
    end
        
end
end
function total_loss = sploss(R, P, Q, a, d, pace, is_hard)
[M,N] = size(R);
total_loss =  sum( sum( (P.' * spdiags(a, 0, M, M) * P) .* (Q.' * spdiags(d, 0, N, N) * Q)));
[I, J, ~] = find(R);
pred = sum(P(I,:).* Q(J,:), 2); loss = pred.^2;
total_loss = total_loss - sum( loss .* a(I) .* d(J));
total_loss = total_loss - 1/pace * (sum(a)*sum(d) - sum(a(I) .* d(J)));
if ~is_hard
    total_loss = 0.5/pace * ( sum(a.^2) * sum(d.^2) - sum( (a(I).^2) .* (d(J).^2) ) );
end
end
function pace = initial_pace(R, P, Q, a, d)
[M, N] = size(R);
total_loss =  sum( sum( (P.' * spdiags(a, 0, M, M) * P) .* (Q.' * spdiags(d, 0, N, N) * Q)));
[I, J, ~] = find(R);
pred = sum(P(I,:).* Q(J,:), 2); loss = pred.^2;
total_loss = total_loss - sum( loss .* a(I) .* d(J));
pace = (M*N - nnz(R)) / total_loss;
end