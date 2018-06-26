function metric = compute_item_metric(mat_rank, user_count, cand_count, topk, cutoff)
if isempty(mat_rank)
    metric = struct();
    return
end
[prec, recall, map] = compute_prec_recall(mat_rank, user_count, cutoff); 
% cand_count column vector storing how many candidate entries for ranking
ndcg = compute_ndcg(mat_rank~=0, mat_rank, cutoff);

if topk>0
    metric = struct('item_recall', recall, 'item_prec', prec, 'item_map', map, 'item_ndcg', ndcg);
else
    
    [auc,mpr] = compute_AUC(mat_rank, user_count, cand_count);
    metric = struct('item_recall', recall, 'item_prec', prec, 'item_map', map, 'item_mpr', mpr, 'item_ndcg', ndcg, 'item_auc', auc);
end

end



