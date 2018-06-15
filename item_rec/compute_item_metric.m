function metric = compute_item_metric(mat_rank, user_count, cand_count, topk, cutoff)
if isempty(mat_rank)
    metric = struct();
    return
end
[M, N] = size(mat_rank);
[user, ~, rank] = find(mat_rank);

rank_mat = sparse(user, rank, 1, M, N);
[prec, recall, map] = PrecRecall(rank_mat, user_count, cutoff); 
% cand_count column vector storing how many candidate entries for ranking

dcg_score = sparse(user, rank, 1./log2(rank+1), M, N);
dcg = full(cumsum(dcg_score(:,1:cutoff), 2));
%idcg = zeros(size(dcg,1),cutoff);
% @todo rel_count should be replaced with the number of items discovered until the position k?
idcg_score = cumsum(1./log2(1+(1:cutoff)));
user_sum_rel = cumsum(rank_mat(:,1:cutoff), 2);
[u, r, c] = find(user_sum_rel);
idcg = full(sparse(u, r, idcg_score(c), size(user_sum_rel,1), size(user_sum_rel,2)));
%for u=1:size(dcg,1)
%    rel_count = min(user_count(u),cutoff);
%    idcg(u, 1:rel_count)= 1./log2(1+(1:rel_count));
%end
%idcg = cumsum(idcg, 2);
%ndcg = full(mean(dcg ./ idcg));
ndcg = dcg./idcg;
ndcg(isnan(ndcg)) = 0;
ndcg = mean(ndcg);

if topk>0
    metric = struct('recall', recall, 'prec', prec, 'map', map, 'ndcg', ndcg);
else
    mat_rank_rank = sparse(user, rank, rank, M, N);
    tmp = sum(mat_rank_rank, 2) - user_count;
    mpr = mean(tmp ./ cand_count ./ user_count);
    auc = ComputeAUC(mat_rank_rank, cand_count, user_count);
    metric = struct('recall', recall, 'prec', prec, 'map', map, 'mpr', mpr, 'ndcg', ndcg, 'auc', auc);
end

end


function AUC = ComputeAUC(user_rank, cand_count, rel_count)
auc_vector = rel_count .* cand_count - sum(user_rank, 2) - (rel_count .* (rel_count - 1))/2;
auc_vector = auc_vector ./ ((cand_count - rel_count) .* rel_count);
AUC = full(mean(auc_vector));
end

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
