function metric = compute_item_metric(test, mat_rank, cand_count, cutoff)
if isempty(mat_rank)
    metric = struct();
    return
end
user_count = sum(test~=0, 2);
istopk = norm(user_count - sum(mat_rank~=0,2))>1e-3;
[prec, recall, map] = compute_prec_recall(mat_rank, user_count, cutoff); 
% cand_count column vector storing how many candidate entries for ranking
ndcg = compute_ndcg(test, mat_rank, cutoff);

if istopk
    metric = struct('item_recall', recall, 'item_prec', prec, 'item_map', map, 'item_ndcg', ndcg);
else
    [auc,mpr] = compute_AUC(mat_rank, user_count, cand_count);
    metric = struct('item_recall', recall, 'item_prec', prec, 'item_map', map, 'item_mpr', mpr, 'item_ndcg', ndcg, 'item_auc', auc);
end

end



