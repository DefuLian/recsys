function R = sample_negative(Rt)
    [N, M] = size(Rt);
    %weight = full(sum(Rt,2));
    weight = ones(N,1);
    items = 1:N;
    user_list = cell(M, 1);
    item_list = cell(M, 1);
    val_list = cell(M, 1);
    parfor i=1:M
        r = Rt(:,i);
        [I,J,v] = find(r);
        w = weight;
        w(I) = 0;
        neg = datasample(items, nnz(r), 'Weight', w, 'Replace', false);
        item_list{i} = [I; neg'];
        val_list{i} = [v; -v];
        user_list{i} = [i*J; i*J];
    end
    R = sparse(cell2mat(user_list), cell2mat(item_list), cell2mat(val_list), M, N);
end