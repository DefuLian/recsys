function [prec, recall] = Evaluate(train, test, P, Q)
[matk, user_count] = PredictK(test, train, P, Q.', 500);
[prec,recall] = PrecRecall(matk, user_count, 500);
end

function [ prec, recall ] = PrecRecall(user_rank, user_count, k )
M = size(user_rank, 1);
user_count_inv = spdiags(1./user_count, 0, M, M);
cum = cumsum(user_rank(:,1:k), 2);
recall = full(mean(user_count_inv * cum));
prec = full(mean(cum * spdiags(1./(1:k)', 0, k, k)));
end

function [mat, user_count, cand_count] = PredictK( R, E, U, Vt, k )
%Predict: Based user and item latent vector, predict items' rank for each user
%   R: test rating matrix, where each entry represents wheather the user
%   has action on corresponding item; E: training rating matrix, sharing
%   similar meaning to R.
%   users(i) has one item with ranks(i)
user_count = sum(R>0 & xor(E>0, R>0), 2);
user_ind = (user_count > 0.001);
%fprintf('%d users are evaluated\n', nnz(user_ind));
R = R(user_ind, :);
E = E(user_ind, :);
U = U(user_ind, :);
user_count = user_count(user_ind);
[M, N] = size(R);
cand_count = N - sum(E > 0, 2);
step = 100;
num_step = floor((M + step-1)/step);
user_cell = cell(num_step, 1);
rank_cell = cell(num_step, 1);
for i=1:num_step
    start_u = (i-1)*step +1;
    end_u = min(i * step, M);
    subU = U(start_u:end_u, :);
    subR = R(start_u:end_u, :);
    subE = E(start_u:end_u, :);
    subR_E = full(subU * Vt);
    subR_E(subE > 0) = -inf;
    [~, Index] = maxk(subR_E, k, 2);
    [I, val, J] = find(Index);
    rank = sparse(I, J, val, size(subR_E,1), size(subR_E,2));
    sub_rank = (subR>0) .* rank;
    [I, ~, val] = find(sub_rank);
    if ~isempty(I)
        user_cell{i} = I + (i-1)*step;
        rank_cell{i} = val;
    end
end
mat = sparse(cell2mat(user_cell), cell2mat(rank_cell), 1, M, N);
end

