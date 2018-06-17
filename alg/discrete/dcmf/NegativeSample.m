function matrix = NegativeSample(matrix, times)
% matrix is a positive only
matrix_t = matrix.';
[M, N] = size(matrix);
cand = (1:N).';
pop = full(sum(matrix));
cell_out = cell(M,3);
for i=1:M
    row = matrix_t(:,i);
    num = nnz(row)*times;
    JN = randsample(cand(~row), num, true, pop(~row));
    IN = i*ones(num,1);
    VN = -ones(num,1); 
    [JP, IP, VP] = find(row);
    cell_out{i,1} = [i*IP; IN]; 
    cell_out{i,2} = [JP; JN]; 
    cell_out{i,3} = [VP; VN];
end
mm = cell2mat(cell_out);
matrix = sparse(mm(:,1), mm(:,2), mm(:,3), M, N);
end