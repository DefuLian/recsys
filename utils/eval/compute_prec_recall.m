function [ prec, recall, map ] = PrecRecall(user_rank, user_count, k )
M = size(user_rank, 1);
user_count_inv = spdiags(1./user_count, 0, M, M);
cum = cumsum(user_rank(:,1:k), 2);
recall = full(mean(user_count_inv * cum));
prec_cum = cum * spdiags(1./(1:k)', 0, k, k);
prec = full(mean(prec_cum));
div = min(repmat(1:k, M, 1), repmat(user_count, 1, k));
map = full(mean(cumsum(prec_cum .* user_rank(:,1:k), 2) ./ div));
end
