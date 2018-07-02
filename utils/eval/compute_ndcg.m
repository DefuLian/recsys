function ndcg = compute_ndcg(test, mat_rank, cutoff)
[M, N] = size(mat_rank);
[user, ~, rank] = find(mat_rank);
%score = test(sub2ind([M,N], user, item));
%assert(all(score>0))
%[~, ~, score] = find(test);
score = test(mat_rank~=0);
dcg_score = sparse(user, rank, score./log2(rank+1), M, N);
dcg = full(cumsum(dcg_score(:,1:cutoff), 2));
dcg = [dcg,full(sum(dcg_score, 2))];
idcg = zeros(size(dcg,1),cutoff+1);
test_t = test';
for u=1:size(dcg,1)
    r = test_t(:,u);
    if nnz(r)==0
        continue;
    end
    r = r(r~=0);
    idcg_ = cumsum(sort(r, 'descend') ./ log2(1+(1:length(r))'));
    if cutoff > length(idcg_)
        idcg(u, :) = [idcg_; repmat(idcg_(end), cutoff+1-length(r),1)];
    else
        idcg(u, :) = [idcg_(1:cutoff); idcg_(end)];
    end
end
ndcg = dcg ./ idcg;
ndcg(isnan(ndcg)) = 0;
ndcg = full(mean(ndcg, 1));
% @todo rel_count should be replaced with the number of items discovered until the position k?
%idcg_score = cumsum(1./log2(1+(1:cutoff)));
%user_sum_rel = cumsum(rank_mat(:,1:cutoff), 2);
%[u, r, c] = find(user_sum_rel);
%idcg = full(sparse(u, r, idcg_score(c), size(user_sum_rel,1), size(user_sum_rel,2)));

%ndcg = dcg./idcg;
%ndcg(isnan(ndcg)) = 0;
%ndcg = mean(ndcg);
end