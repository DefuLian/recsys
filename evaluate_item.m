function evalout = evaluate_item(train, test, P, Q, topk, cutoff)
% user_count column vector storing how many entries in the test 
if nnz(test)>0
    test(train~=0) = 0;
    user_count = sum(test~=0,2);
    idx = user_count > 0.0001; train = train(idx,:); test = test(idx,:); P = P(idx,:);
    [mat_rank, ~, cand_count] = Predict(test, train, P, Q, topk);
    if isexplict(test)
        evalout = compute_rating_metric(test, mat_rank, cand_count, cutoff);
    else
        evalout = compute_item_metric(test, mat_rank, cand_count, cutoff);
    end
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
%if size(test, 2)==3 && sum(test(:,3)==0) > sum(test(:,3)~=0)
%    [ mat, user_count, cand_count ] = Predict_Tuple(test, train, U, V);
%else
    Et = train.';
    Rt = (test~=0).';
    Ut = U.';
    user_count = sum(Rt~=0 & xor(Et~=0, Rt~=0));
    %if nnz(test)>0
    %    Ind = (user_count > 0.0001);
    %    Rt = Rt(:, Ind);
    %    Et = Et(:, Ind);
    %    Ut = Ut(:, Ind);
    %    user_count = user_count(Ind);
    %end
    [N, M] = size(Rt);
    cand_count = N - sum(Et ~= 0);
    step = 100;
    num_step = floor((M + step-1)/step);
    user_cell = cell(num_step, 1);
    rank_cell = cell(num_step, 1);
    item_cell = cell(num_step, 1);
    if nargout == 4
        topk_cell = cell(num_step, 1);
    end
    [Utc, Vc] = split_latent_matrix(Ut,V);
    for i=1:num_step
        start_u = (i-1)*step +1;
        end_u = min(i * step, M);
        %subU = Ut(:, start_u:end_u); %%
        subUc = get_subUc(Utc, start_u, end_u);
        subR = Rt(:, start_u:end_u);
        subE = Et(:, start_u:end_u);
        %subR_E1 = full(V * subU); %%
        subR_E = multiply_cell(subUc, Vc);
        subR_E(subE ~= 0) = -inf;
        if topk>0
            [score, Index] = maxk(subR_E, topk);
            [val, I, J] = find(Index);
            rank = sparse(J, I, val, size(subR_E,1), size(subR_E,2));
            if nargout == 4
                %topk_cell{i} = [I + (i-1)*step, J, val];
                topk_cell{i} = [I + (i-1)*step, J, score(:)];
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
%end

function [Utc, Vc] = split_latent_matrix(Ut,V)
M = size(Ut,2);
N = size(V,1);
urows = sum(Ut~=0, 2);
dense_u_cols = urows ./M > 0.5;
vrows = sum(V~=0);
dense_v_cols = vrows ./N > 0.5;
cols = dense_u_cols | dense_v_cols.';
Utc = {full(Ut(cols,:)),Ut(~cols,:)};
Vc = {full(V(:,cols)),V(:,~cols)};
end
function subUtc = get_subUc(Utc, starti, endi)
subUtc = cell(length(Utc),1);
for i=1:length(Utc)
    subUtc{i} = Utc{i}(:, starti:endi);
end
end
function mat = multiply_cell(Utc, Vc)

for i=1:length(Utc)
    if i ==1
        mat = Vc{i} * Utc{i};
        %mat = mult_real(Vc{i}, Utc{i}');
    else
        mat = mat + Vc{i} * Utc{i};
        %mat = mat + mult_real(Vc{i}, Utc{i}');
    end
end
end




