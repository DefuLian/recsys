function [ mat, user_count, cand_count ] = predict_tuple(topkmat, test, U, V, cutoff)
%Predict: Based user and item latent vector, predict items' rank for each
%user, but candicate items are given in R
%   R: test data with three columns, user column, item column and relevance
%   column; E: training rating matrix
%   users(i) has one item with ranks(i)
Vt = V.';
I = topkmat(:,1);
J = topkmat(:,2);
%Val = topkmat(:,3);
cand_count = tabulate(I); cand_count = cand_count(:,2);
user_count = sum(test~=0, 2);
cum_cand_count = [0;cumsum(cand_count)];
user_cell = cell(length(cand_count), 1);
rank_cell = cell(length(cand_count), 1);
item_cell = cell(length(cand_count), 1);
[M,N] = size(test);
test_t = test';
for u=1:length(cand_count)
    idx_start = cum_cand_count(u)+1;
    idx_end = cum_cand_count(u+1);
    if idx_start > idx_end
        continue
    end
    r = test_t(:,u);
    cand_items = J(idx_start:idx_end);
    pred = U(u,:) * Vt(:,cand_items) ;%+ Val(idx_start:idx_end)';
    [~, idx] = maxk(pred, cutoff, 2, 'sorting', true);
    items = cand_items(idx);
    users = ones(length(items),1) * u;
    rank = (1:length(items))';
    real_items = find(r);
    [~, idx] = intersect(items, real_items);
    user_cell{u} = users(idx);
    item_cell{u} = items(idx);
    rank_cell{u} = rank(idx);
end
mat = sparse(cell2mat(user_cell), cell2mat(item_cell), cell2mat(rank_cell), M, N);
%idx = user_count > 0.001;
%mat = mat(idx, :); user_count = user_count(idx); cand_count = cand_count(idx);
end
