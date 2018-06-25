function metric = compute_item_metric(mat_rank, user_count, cand_count, topk, cutoff)
if isempty(mat_rank)
    metric = struct();
    return
end
[M, N] = size(mat_rank);
[user, ~, rank] = find(mat_rank);

rank_mat = sparse(user, rank, 1, M, N);
[prec, recall, map] = compute_prec_recall(rank_mat, user_count, cutoff); 
% cand_count column vector storing how many candidate entries for ranking
ndcg = compute_ndcg(mat_rank~=0, mat_rank, cutoff);

if topk>0
    metric = struct('item_recall', recall, 'item_prec', prec, 'item_map', map, 'item_ndcg', ndcg);
else
    mat_rank_rank = sparse(user, rank, rank, M, N);
    tmp = sum(mat_rank_rank, 2) - user_count;
    mpr = mean(tmp ./ cand_count ./ user_count);
    auc = compute_AUC(mat_rank_rank, cand_count, user_count);
    metric = struct('item_recall', recall, 'item_prec', prec, 'item_map', map, 'item_mpr', mpr, 'item_ndcg', ndcg, 'item_auc', auc);
end

end



