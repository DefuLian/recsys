function [U, V] = irenmf(R, varargin)
    R = +(R>0);
    [M, N] = size(R);
    [max_iter, K, item_sim, reg_u, reg_i, reg_s, tol, alpha, itemGroup] = ...
    process_options(varargin, 'max_iter', 50, 'K', 150, 'item_sim', sparse(N,N), 'reg_u', 0.015, ...
    'reg_i', 0.015, 'reg_s', 1, 'tol', 1e-4, 'alpha', 30, 'itemGroup', []);
    if isempty(itemGroup)
        error('please provide item group data\n');
    end
    
    W = R * alpha;
    geo_alpha = 0.6;
    %rng(10);
    userW=sqrt(1/K)*rand(M, K+2);  
    itemW=sqrt(1/K)*rand(N, K+2); 
    userW(:, 1)=1; 
    itemW(:, end)=1;
    LastError=0;
    item_sim = geo_alpha * spdiags(ones(N,1), 0, N, N) + (1 - geo_alpha) * item_sim;
    Rt = R.';
    Wt = W.';
    for e = 1 : max_iter
        tic;
        %userW= APGUserLatentFactor(W, R, userW, item_sim*itemW, reg_u);        
        userW = Optimize(Rt, Wt, userW, item_sim*itemW, reg_u);
        itemW= APGItemLatentFactor(W, R, userW, itemW, reg_i, reg_s, item_sim, itemGroup);   
        CurrError = 0.5*fast_loss(R,W,userW, item_sim*itemW);
        deltaError=(CurrError - LastError)/abs(LastError);
        fprintf('Epoch %g, CurrError %g, LastError %g, DeltaErr %g, Time: %g\n', e, CurrError, LastError, deltaError, toc);
        %if abs(deltaError) < tol            
        %    break;
        %end
        LastError=CurrError;        
    end
    U = userW; V = item_sim * itemW;
    fprintf('complete the learning of the model parameters\n');
    %--------------- the sparsity of latent factors of item------------------------%
    fprintf('num_factors:%g, lambda:%g %g %g, ', K, reg_u, reg_i, reg_s);
    density=nnz(itemW)/numel(itemW);    
    fprintf('item latent factors density:%g\n', density);

end


function U = Optimize(R, W, U, V, reg)
[M, K] = size(U);
Vt = V.';
VtV = Vt * V + reg * eye(K);
for i = 1 : M
    w = W(:, i);
    r = R(:, i);
    Ind = w>0;
    if nnz(w) == 0
        Wi = zeros(0);
    else
        Wi = diag(w(Ind));
    end
    sub_V = V(Ind,:);
    VCV = sub_V.' * Wi * sub_V + VtV; %Vt_minus_V = sub_V.' * (Wi .* sub_V) + invariant;
    Y = Vt * (w .* r + r);
    u = VCV \ Y;
    U(i,:) = u;
end
end

