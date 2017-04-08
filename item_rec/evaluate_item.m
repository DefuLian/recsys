function evalout = evaluate_item(train, test, P, Q, topk, cutoff)

% user_count column vector storing how many entries in the test  
if nnz(test)>0
    [mat_rank, user_count, cand_count] = Predict(test, train, P, Q, topk);
    evalout = compute_item_metric(mat_rank, user_count, cand_count, topk, cutoff);
else
    if topk<=0
        error('Please give positive topk value')
    end
    [~, ~, ~, evalout] = Predict(test, train, P, Q, topk);
end
end


function [ mat, user_count, cand_count, topkmat ] = Predict(test, train, U, V, topk )
%Predict: Based user and item latent vector, predict items' rank for each
%user
%   R: test rating matrix, where each entry represents wheather the user
%   has action on corresponding item; E: training rating matrix, sharing
%   similar meaning to R.
%   users(i) has one item with ranks(i)
if size(test, 2)==3 && sum(test(:,3)==0) > sum(test(:,3)>0)
    [ mat, user_count, cand_count ] = Predict_Tuple(test, train, U, V);
else
    Et = train.';
    Rt = (test>0).';
    Ut = U.';
    user_count = sum(Rt~=0 & xor(Et~=0, Rt~=0));
    if nnz(test)>0
        Ind = (user_count > 0.0001);
        Rt = Rt(:, Ind);
        Et = Et(:, Ind);
        Ut = Ut(:, Ind);
        user_count = user_count(Ind);
    end
    [N, M] = size(Rt);
    cand_count = N - sum(Et > 0);
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
    cand_count = cand_count.';
    user_count = user_count.';
end
end


function [ mat, user_count, cand_count ] = Predict_Tuple(R, E, U, V )
%Predict: Based user and item latent vector, predict items' rank for each
%user, but candicate items are given in R
%   R: test data with three columns, user column, item column and relevance
%   column; E: training rating matrix
%   users(i) has one item with ranks(i)
Vt = V.';
Et = E.';
I = R(:,1);
J = R(:,2);
Val = R(:,3);
cand_count = tabulate(I); cand_count = cand_count(:,2);
user_count = tabulate(I(Val>0)); user_count = user_count(:,2);
user_ind = cand_count>0;
cand_count = cand_count(user_ind);
user_count = user_count(user_ind);
U = U(user_ind,:);
Et = Et(:, user_ind);
cum_cand_count = cumsum(cand_count);
cum_cand_count = [0;cum_cand_count];
user_cell = cell(length(cand_count), 1);
rank_cell = cell(length(cand_count), 1);
item_cell = cell(length(cand_count), 1);
M = length(cand_count);
N = max(J);
for u=1:length(cand_count)
    eu = Et(:,u);
    u_start = cum_cand_count(u)+1;
    u_end = cum_cand_count(u+1);
    assert(std(I(u_start:u_end))==0)
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




