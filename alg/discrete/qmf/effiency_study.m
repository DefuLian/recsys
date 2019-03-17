%run('D:/code/recsys/setup.m')
run('~/software/recsys/setup.m')
dataset = 'lastfm';
dir = '~';
output_file = sprintf('%s/result_qcf/%sdata_effiency.mat', dir, dataset);
if ~exist(output_file, 'file')
    load(sprintf('%s/data/%sdata.mat', dir, dataset));

    [train, test] = split_matrix(data, 'un', 0.8);
    fprintf('train=%d, test=%d\n', nnz(train), nnz(test));
    if isexplict(data)
        load(sprintf('%s/result_qcf/%sdata_result_qcf.mat', dir, dataset))
    else
        load(sprintf('%s/result_qcf/%sdata_result_if_qcf.mat', dir, dataset))
    end

    iccf_para = result{1}{3};

    %if isexplict(data)
    %    load(sprintf('%s/result_qcf/%sdata_result_dmf.mat', dir, dataset))
    %else
    %    load(sprintf('%s/result_qcf/%sdata_result_if_dmf.mat', dir, dataset))
    %end
    %dmf_para = result{end}{3};

    K=64;
    [B,D]=dmf(train, 'K', K, 'max_iter',20, 'rho', 1/iccf_para{2}, 'alpha', 0, 'beta', 0);
    [P,Q] = iccf(train, 'max_iter', 20, 'K', K, iccf_para{:}); 
    P = P(:,1:end-2); Q = Q(:,1:end-2);
    [~,~,X,Y,U,V] = qcf(train, 'max_iter', 20, 'K', K, iccf_para{:}); 
    %[B,D]=dmf(train, 'K', K, 'max_iter',20, dmf_para{:});
    

    save(output_file, 'train', 'test', 'P', 'Q', 'X', 'Y', 'U', 'V', 'B', 'D')
    clear('result');
else
    load(output_file);
end

%if ~exist('result','var')
    [M, N] = size(train);
    can_num = 10;
    max_iter = 5;
    time = zeros(3*length(can_num)+1, max_iter);
    result = cell(3*length(can_num)+1, max_iter);
    for iter=1:max_iter
        if M > 50000
            idx = randi(M, 50000, 1);
            train1 = train(idx,:); test1 = test(idx,:);
            P1 = P(idx,:); X1 = X(idx,:); B1 = B(idx,:);
        else
            train1 = train; test1 = test;
            P1 = P; X1 = X; B1 = B;
        end
        
        for i=1:can_num
            fprintf('iter=%d,i=%d\n', iter, i);
            [result{i, iter}, time(i, iter)] = mip_search(struct('real', P1, 'code', X1, 'word', U), struct('real', Q, 'code', Y, 'word', V), train1, test1, 200*i);
        end
        for i=1:can_num
            fprintf('iter=%d,i=%d\n', iter, i+can_num);
            [result{i+can_num, iter}, time(i+can_num, iter)] = mip_search(struct('real', P1, 'code', B1), struct('real', Q, 'code', D), train1, test1, 200*i);
        end
        for i=1:can_num
            fprintf('iter=%d,i=%d\n', iter, i+can_num*2);
            [result{i+can_num*2, iter}, time(i+can_num*2, iter)] = mip_search(P1, Q, train1, test1, 200, size(P,2), floor(N*(0.05+0.01*i)));
        end
        [result{31, iter}, time(31, iter)] = mip_search(P1, Q, train1, test1, 200);
    end
    
    save(output_file, 'result', 'time', '-append');

%end