function [data, rows, cols] = trim_data(data, count)
if nargin == 1
    count = 10;
end
[M, N] = size(data);
cols = 1:N;
rows = 1:M;
while(true)
    col_sum = sum(data~=0);
    col_ind = col_sum>=count;
    if nnz(col_ind)>0
        cols = cols(col_ind);
        data = data(:,col_ind);
    end
    row_sum = sum(data~=0,2);
    row_ind = row_sum>=20;
    if nnz(row_ind)>0
        rows = rows(row_ind);
        data = data(row_ind,:);
    end
    if nnz(~col_ind) == 0 && nnz(~row_ind) == 0
        break;
    else
        fprintf('%d\t%d\n', nnz(col_ind), nnz(row_ind));
    end
end
end
