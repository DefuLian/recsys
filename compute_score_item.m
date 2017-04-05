function [metric, topkmat] = compute_score_item(train, test, P, Q, topk, cutoff)

% user_count column vector storing how many entries in the test

if nargin ==5
    cutoff = 100;
end
    
if topk>0 && topk < cutoff
    topk = cutoff;
end

if nnz(test)>0
    [mat_rank, user_count, cand_count] = Predict(test, train, P, Q, topk);
else
    if topk<=0
        error('Please give positive topk value')
    end
    [~, ~, ~, topkmat] = Predict(test, train, P, Q, topk);
    metric = struct();
    return
end
[M, N] = size(mat_rank);
[user, ~, rank] = find(mat_rank);


[prec, recall, map] = PrecRecall(sparse(user, rank, 1, M, N), user_count, cutoff); 
% cand_count column vector storing how many candidate entries for ranking

dcg_score = sparse(user, rank, 1./log2(rank+1), M, N);
dcg = cumsum(dcg_score(:,1:cutoff), 2);
idcg = zeros(size(dcg,1),cutoff);
for u=1:size(dcg,1)
    rel_count = min(user_count(u),cutoff);
    idcg(u, 1:rel_count)= 1./log2(1+(1:rel_count));
end
idcg = cumsum(idcg, 2);
ndcg = mean(dcg ./ idcg);


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
AUC = mean(auc_vector);
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


function [ mat, user_count, cand_count, topkmat ] = Predict( R, E, U, V, topk )
%Predict: Based user and item latent vector, predict items' rank for each
%user
%   R: test rating matrix, where each entry represents wheather the user
%   has action on corresponding item; E: training rating matrix, sharing
%   similar meaning to R.
%   users(i) has one item with ranks(i)

if size(R, 2)==3 && sum(R(:,3)==0) > sum(R(:,3)>0)
    [ mat, user_count, cand_count ] = Predict_Tuple(R, E, U, V);
else
    Et = E.';
    Rt = (R>0).';
    Ut = U.';
    user_count = sum(R~=0 & xor(E~=0, R~=0), 2);
    %Ind = (user_count > 0.001);
    %Rt = Rt(:, Ind);
    %Et = Et(:, Ind);
    %Ut = Ut(:, Ind);
    %user_count = user_count(Ind);
    [M, N] = size(R);
    cand_count = N - sum(E > 0, 2);
    step = 100;
    num_step = floor((M + step-1)/step);
    user_cell = cell(num_step, 1);
    rank_cell = cell(num_step, 1);
    item_cell = cell(num_step, 1);
    if nargout == 4
        topk_cell = cell(num_step, 1);
    end
    for i=1:num_step
        start_u = (i-1)*step +1;
        end_u = min(i * step, M);
        subU = Ut(:, start_u:end_u);
        subR = Rt(:, start_u:end_u);
        subE = Et(:, start_u:end_u);
        subR_E = full(V * subU);
        subR_E(subE > 0) = -inf;
        if topk>0
            [~, Index] = maxk(subR_E, topk);
            [val, I, J] = find(Index);
            rank = sparse(J, I, val, size(subR_E,1), size(subR_E,2));
            if nargout == 4
                topk_cell{i} = [I + (i-1)*step, J, val];
            end
        else
            [~, Index] = sort(subR_E, 'descend');
            [~, rank] = sort(Index);
            
        end
        sub_rank = subR .* rank;
        [J, I, val] = find(sub_rank);
        if ~isempty(I)
            user_cell{i} = I + (i-1)*step;
            rank_cell{i} = val;
            item_cell{i} = J;
        end
    end
    mat = sparse(cell2mat(user_cell), cell2mat(item_cell), cell2mat(rank_cell), M, N);
    if nargout == 4
        topkmat = cell2mat(topk_cell);
    end
end
end


function [ mat, user_count, cand_count ] = Predict_Tuple(R, E, U, V )
%Predict: Based user and item latent vector, predict items' rank for each
%user, but candicate items are given in R
%   R: test data with three columns, user column, item column and relevance
%   column; E: training rating matrix
%   users(i) has one item with ranks(i)
Vt = V;
Et = E;
I = R(:,1);
J = R(:,2);
Val = R(:,3);
cand_count = crosstab(I);
user_count = crosstab(I(Val>0));
cum_user_count = cumsum(user_count);
cum_user_count = [0;cum_user_count];
user_cell = cell(length(cand_count), 1);
rank_cell = cell(length(cand_count), 1);
item_cell = cell(length(cand_count), 1);
for u=1:length(cand_count)
    if cand_count(u)==0
        continue;
    end
    eu = Et(:,u);
    u_start = cum_user_count(u)+1;
    u_end = cum_user_count(u+1);
    cand_items = J(u_start:u_end);
    pred = U(u,:) * Vt(:,cand_items);
    cols = [pred.', Val(u_start:u_end),cand_items];
    e_ind = eu(cand_items)==0;
    cand_count(u) = nnz(e_ind);
    cols = cols(e_ind, :);
    cols_sorted = sortrows(cols, -1);
    rel_rank = find(cols_sorted(:,2));
    user_cell{u} = u * ones(length(rel_rank),1);
    rank_cell{u} = rel_rank;
    item_cell{u} = cols_sorted(rel_rank,3);
end
mat = sparse(cell2mat(user_cell), cell2mat(item_cell), cell2mat(rank_cell), M, N);
end




