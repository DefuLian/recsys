function eval = compute_rating_metric(test, mat_rank, cand_count, cutoff)
test1 = explicit2implicit(test);
idx = test1~=0 & mat_rank~=0; [I,J] = find(idx); [m,n] = size(test);
eval_like = compute_item_metric(test1, sparse(I,J,mat_rank(idx),m,n), cand_count, cutoff);

test2 = test~=0;
eval_view = compute_item_metric(test2, mat_rank, cand_count, cutoff);

names = [cellfun(@(x) sprintf('%s_like', x), fieldnames(eval_like), 'UniformOutput',false); ...
    cellfun(@(x) sprintf('%s_view',x), fieldnames(eval_view), 'UniformOutput',false)];
eval = cell2struct([struct2cell(eval_like); struct2cell(eval_view)], names, 1);

eval.item_ndcg_score = compute_ndcg(test, mat_rank, cutoff);
end