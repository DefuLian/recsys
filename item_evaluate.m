function [recall, prec, mpr] = item_evaluate(train, test, P, Q)
[mat_rank, user_count, cand_count] = Predict(test, train, P, Q.', 100);
[prec,recall] = PrecRecall(mat_rank, user_count, 100);
tmp = sum(mat_rank, 2) - user_count;
mpr = mean(tmp ./ cand_count ./ user_count);
end
function [ prec, recall ] = PrecRecall(user_rank, user_count, k )
M = size(user_rank, 1);
user_count_inv = spdiags(1./user_count, 0, M, M);
cum = cumsum(user_rank(:,1:k), 2);
recall = full(mean(user_count_inv * cum));
prec = full(mean(cum * spdiags(1./(1:k)', 0, k, k)));
end
function [ mat, user_count, cand_count ] = Predict( R, E, U, Vt )
%Predict: Based user and item latent vector, predict items' rank for each
%user
%   R: test rating matrix, where each entry represents wheather the user
%   has action on corresponding item; E: training rating matrix, sharing
%   similar meaning to R.
%   users(i) has one item with ranks(i)
user_count = sum(R>0 & xor(E>0, R>0), 2);
Ind = (user_count > 0.1);
R = R(Ind, :);
E = E(Ind, :);
U = U(Ind, :);
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
    [~, Index] = sort(subR_E, 2, 'descend');
    [~, rank] = sort(Index, 2);
    sub_rank = (subR>0) .* rank;
    [I, ~, val] = find(sub_rank);
    user_cell{i} = I + (i-1)*step;
    rank_cell{i} = val;
end
mat = sparse(cell2mat(user_cell), cell2mat(rank_cell), 1, M, N);
end

