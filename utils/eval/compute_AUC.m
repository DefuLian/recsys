function [AUC,mpr] = compute_AUC(mat_rank, rel_count, cand_count)
tmp = sum(mat_rank, 2) - rel_count;
mpr = mean(tmp ./ cand_count ./ rel_count);
auc_vector = rel_count .* cand_count - sum(mat_rank, 2) - (rel_count .* (rel_count - 1))/2;
auc_vector = auc_vector ./ ((cand_count - rel_count) .* rel_count);
AUC = full(mean(auc_vector));
end