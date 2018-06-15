function [P, Q] = splicf(R, varargin)
    [beta, test, max_iter, K, reg_u, reg_i, spr] = process_options(varargin, 'beta', 1/30, 'test', [], 'max_iter', 8, 'K', 50, ...
        'reg_u', 1, 'reg_i', 1, 'spr', 'lin_soft');
    [M, N] = size(R); eta = .9; inner_max_iter = 20;
    rng(100);
    Q = randn(N, K) * 0.1; P = randn(M, K);
    %W = get_weight(R, P, Q, eta, spr, 0);
    %W = sparse(M, N); 
    ratios = [0.7, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1];
    for iter=1:(max_iter)
        W = get_weight(R, P, Q, eta, spr, ratios(iter));
        fprintf('outer iteration #=%3d of SPL\n', iter);
        [P, Q] = optimize(R, W, P, Q, inner_max_iter, beta, reg_u, reg_i, test);
        %Wp = W; %Wn = addw(W,Wp);
        %fprintf('droped %d entries\n', (nnz(Wn)-nnz(W))/nnz(W));
        %W = Wn;
    end
    
end

function [P, Q] = optimize(R, W, P, Q, max_iter, beta, reg_u, reg_i, test)
    Wt = W.';
    Rt = R.';
    for inner_iter=1:max_iter
        P = optimize_newton(Rt, Q, Wt, beta, reg_u);
        Q = optimize_newton(R, P, W, beta, reg_i);
        fprintf('    inner #iteration=%d, loss=%f, mean_c_loss=%f\n', inner_iter, ...
            compute_total_loss(R, P, Q, beta), compute_classification_loss(R,P,Q));
    end
    eval = evaluate_item(R, test, P, Q, -1, 200);
    fprintf('recall@50=%f, ndcg@50=%f, auc=%f\n', eval.recall(50), eval.ndcg(50), eval.auc());
end

function P = optimize_newton(Rt, Q, Wt, beta, reg)
    [~, M] = size(Rt);
    K = size(Q, 2);
    P = zeros(M, K);
    QtQ = Q.' * Q;
    parfor i=1:M 
        %fprintf('user=%d\n', i);
        col_r = Rt(:,i);
        col_w = Wt(:,i);
        p = zeros(K,1); pp = p;
        ind = (col_w > eps);
        w = col_w(ind);
        X = Q(ind, :);
        y = (col_r(ind) + 1) / 2; % convert {+1,-1} to {1, 0}
        XtX = 2 * beta * (QtQ - X.' * X) + 2 * reg * eye(K);
        for iter = 1:50
            mu = 1./(1 + exp(-X * p));
            s = w .* mu .* (1 - mu);
            XsX = X.' * spdiags(s, 0, length(s), length(s)) * X;
            g = X.' * (w .* (mu - y)) +  XtX * p;
            H = XsX  + XtX;
            d = H \ g;
            p = p - d;
            if norm(p - pp) < 1e-3
                break
            end
            pp = p; 
        end
        P(i,:) = p;
    end
end


function P = optimize_jj_bound(Rt, Q, Wt, beta)
    [N, M] = size(Rt);
    for i=1:M
        
    end
end
function P = optimize_bohning_bound(Rt, Q, Wt, beta)
end

function W = optimize_weight(R, P, Q, k_small_array, k_large_array, choice)
    classes = [-1, 1];
    [M, N] = size(R);
    I_list = cell(2,1);
    J_list = cell(2,1);
    w_list = cell(2,1);
    ww = [0.1, 1];
    for i=1:2
        k_small = k_small_array(i);
        k_large = k_large_array(i);
        class = classes(i);
        [I, J, y] = find(R == class);
        y_hat = sum(P(I,:) .* Q(J,:), 2);
        %loss = log(1+ exp(- y .* y_hat));
        loss = logitloss(y.*y_hat);
        w = zeros(length(y), 1);
        tiny_index = loss < 1/k_large;
        small_index = loss < 1/k_small;
        w(tiny_index) = 1;
        if strcmp(choice, 'lin_soft')
            % w = b - a x
            a = k_large * k_small / (k_large - k_small); b = k_large / (k_large - k_small);
            w(small_index & ~tiny_index) = b - a * loss(small_index & ~tiny_index);
        elseif strcmp(choice, 'mixture_soft')
            % w = a/x - b
            a = 1/(k_large-k_small); b = k_small / (k_large - k_small);
            w(small_index & ~tiny_index) = a ./ loss(small_index & ~tiny_index) - b;
        elseif strcmp(choice, 'sqrt_soft')
            % w = a/sqrt(l) - b
            a = 1/(sqrt(k_large) - sqrt(k_small)); b = sqrt(k_small)/(sqrt(k_large) - sqrt(k_small));
            w(small_index & ~tiny_index) = a ./ sqrt(loss(small_index & ~tiny_index)) - b;
        end
        I_list{i} = I;
        J_list{i} = J;
        w_list{i} = w * ww(i);
    end
    W = sparse(cell2mat(I_list), cell2mat(J_list), cell2mat(w_list), M, N);
end
function [k_small, k_large] = compute_pace(R, P, Q, eta, q)
    k_small = zeros(2,1); k_large = zeros(2,1);
    [I, J, y] = find(R == -1);
    y_hat = sum(P(I,:) .* Q(J, :), 2);
    %loss = log(1+exp(-y .* y_hat));
    loss = logitloss(y .* y_hat);
    %k_small(1) = 1/quantile(loss, q);
    %k_large(1) = 1/quantile(loss, q*eta);
    k_small(1) = 1/min(loss);
    k_large(1) = k_small(1);
    
    [I, J, y] = find(R == 1);
    y_hat = sum(P(I,:) .* Q(J, :), 2);
    %loss = log(1+exp(-y .* y_hat));
    loss = logitloss(y .* y_hat);
    k_small(2) = 1/quantile(loss, q);
    k_large(2) = 1/quantile(loss, q*eta);
    %k_small(2) = 1/(max(loss)+1);
    %k_large(2) = k_small(2);
end
function total_loss = compute_total_loss(R, P, Q, beta)
    [I, J, y] = find(R);
    y_hat = sum(P(I,:) .* Q(J,:),2);
    %loss = log(1+exp(-y.*y_hat));
    loss = logitloss(y.*y_hat);
    total_loss = sum(loss);
    total_loss = total_loss + beta*(sum(sum((P.' * P) .* (Q.' * Q))) - sum(y_hat.^2));
end
function cm_loss = compute_classification_loss(R, P, Q)
    [I, J, y] = find(R==1);
    y_hat = sum(P(I,:) .* Q(J,:),2);
    %loss = log(1+exp(-y.*y_hat));
    loss = logitloss(y.*y_hat);
    cm_loss = mean(loss);
    
end
function str = stat(W, R)
    pos_ind = R>0;
    neg_ind = R<0;
    %[I, J, ~] = find(pos_ind);
    %pos_w = full(W(sub2ind([M, N], I, J))); 
    %[I, J, ~] = find(neg_ind);
    %neg_w = full(W(sub2ind([M, N], I, J)));
    pos_w = full(W(pos_ind));
    neg_w = full(W(neg_ind));
    
    pos_ratio = sum(pos_w>0) / length(pos_w);
    neg_ratio = sum(neg_w>0) / length(neg_w);
    
    pos_qr = quantile(pos_w, [0, 0.25, 0.5, 0.75, 1]);
    neg_qr = quantile(neg_w(neg_w>0), [0, 0.25, 0.5, 0.75, 1]);
    
    pos_1_ratio = sum(pos_w > 0.99) / length(pos_w);
    neg_1_ratio = sum(neg_w > 0.99) / length(neg_w);
    
    str1 = sprintf('    pos_ratio=%.2f, neg_ratio=%.2f\n    pos_1_ratio=%.2f, neg_1_ratio=%.2f', pos_ratio, neg_ratio, pos_1_ratio, neg_1_ratio);
    str = sprintf('%s\n    [%s],[%s]', str1, sprintf('%.2f,', pos_qr), sprintf('%.2f,', neg_qr));
end
function Wn = addw(W, Wp)
    [M,N]= size(W);
    [I, J, V] = find(W);
    [I2, J2, ~] = find(Wp>0 & xor(Wp>0, W>0));
    V2 = full(Wp(sub2ind([M, N], I2, J2)));
    Wn = sparse([I;I2], [J;J2], [V;V2]); 
end

function W = get_weight(R, P, Q, eta, spr, ratio)
    [k_small, k_large] = compute_pace(R, P, Q, eta, ratio);
    W = optimize_weight(R, P, Q, k_small, k_large, spr);
    fprintf('    k_small=[%.2f,%.2f], k_large=[%.2f,%.2f]\n', ...
        k_small(1), k_small(2), k_large(1),k_large(2));
    fprintf('%s\n', stat(W,R));
end

