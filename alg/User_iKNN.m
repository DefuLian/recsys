function [sim, Q] = User_iKNN(train, varargin)
K = process_options(varargin, 'K', inf);
if isinf(K)
    sim = User_Similarity(train);
else
    sim = Compute_Similarity(train, 5000, K);
end
ind = sub2ind(size(sim), ones(size(sim, 1), 1), ones(size(sim, 1), 1));
sim(ind) = 0;
row_sum = sum(sim,2)+eps;
sim = spdiags(1./row_sum, 0, size(sim,1), size(sim,2)) * sim;
Q = +(train>0).';
end

function sim = Compute_Similarity(train, nUserInGroup, K)
R = +(train>0);
[M,~] = size(R);
traint = R.';
nGroups = floor((M + nUserInGroup-1)/nUserInGroup);
nnz_row = sum(R,2);
sim_cell = cell(nGroups,1);
for i=1:nGroups
    start_u = (i-1)*nUserInGroup +1;
    end_u = min(i * nUserInGroup, M);
    sub_mat = R(start_u:end_u, :);
    join = sub_mat * traint;
    [I,J,V] = find(join);
    join = sparse(I,J,V ./ (nnz_row(I)+nnz_row(J)-V), size(join,1), size(join,2));
    [~,ind_mat] = maxk(join, K, 2);
    [I,~,V] = find(ind_mat);
    sim_cell{i} = sparse(I,V,join(sub2ind(size(join), I, V)), size(join,1),size(join,2));
end
sim = cell2mat(sim_cell);
end
function sim = User_Similarity(train)
[M,~] = size(train);
R = +(train>0);
nnz_row = sum(R,2);
join = R * R';
[I,J,V] = find(join);
VV = V./(nnz_row(I) + nnz_row(J) - V);
sim = sparse(I,J,VV, M, M);
end