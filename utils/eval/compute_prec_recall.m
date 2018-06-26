function [ prec, recall, map ] = compute_prec_recall(mat_rank, user_count, cutoff )
[M, N] = size(mat_rank);
[user, ~, rank] = find(mat_rank);
user_rank = sparse(user, rank, 1, M, N);
user_count_inv = spdiags(1./user_count, 0, M, M);
cum = cumsum(user_rank(:,1:cutoff), 2);
recall = full(mean(user_count_inv * cum, 1));
prec_cum = cum * spdiags(1./(1:cutoff)', 0, cutoff, cutoff);
prec = full(mean(prec_cum, 1));
div = min(repmat(1:cutoff, M, 1), repmat(user_count, 1, cutoff));
map = full(mean(cumsum(prec_cum .* user_rank(:,1:cutoff), 2) ./ div, 1));
end
