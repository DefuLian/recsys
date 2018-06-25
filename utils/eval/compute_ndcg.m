function ndcg = compute_ndcg(test, mat_rank, cutoff)
[M, N] = size(mat_rank);
[user, ~, rank] = find(mat_rank);
[~, ~, score] = find(test);
dcg_score = sparse(user, rank, score./log2(rank+1), M, N);
dcg = full(cumsum(dcg_score(:,1:cutoff), 2));
dcg = [dcg,full(sum(dcg_score, 2))];
idcg = zeros(size(dcg,1),cutoff+1);
% @todo rel_count should be replaced with the number of items discovered until the position k?
%idcg_score = cumsum(1./log2(1+(1:cutoff)));
%user_sum_rel = cumsum(rank_mat(:,1:cutoff), 2);
%[u, r, c] = find(user_sum_rel);
%idcg = full(sparse(u, r, idcg_score(c), size(user_sum_rel,1), size(user_sum_rel,2)));
test_t = test';
for u=1:size(dcg,1)
    r = test_t(:,u);
    r = r(r~=0);
    idcg_ = cumsum(sort(r, 'descend') ./ log(1+(1:length(r))'));
    if cutoff > length(idcg_)
        idcg(u, :) = [idcg_; repmat(idcg_(end), cutoff+1-length(r),1)];
    else
        idcg(u, :) = [idcg_(1:cutoff); idcg_(end)];
    end
end
ndcg = full(mean(dcg ./ idcg));
%ndcg = dcg./idcg;
%ndcg(isnan(ndcg)) = 0;
%ndcg = mean(ndcg);
end