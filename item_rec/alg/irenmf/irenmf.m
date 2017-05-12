function [U, V] = irenmf(R, varargin)
    R = +(R>0);
    [M, N] = size(R);
    [max_iter, K, item_sim, reg_u, reg_i, reg_s, tol, alpha, itemGroup] = ...
    process_options(varargin, 'max_iter', 50, 'K', 1e-3, 'item_sim', sparse(N,N), 'reg_u', 0.015, ...
    'reg_i', 0.015, 'reg_s', 1, 'tol', 1e-5, 'alpha', 30, 'itemGroup', []);
    if isempty(itemGroup)
        error('please provide item group data\n');
    end
    
    W = R * alpha;
    geo_alpha = 0.6;
    rng(10);
    userW=sqrt(1/K)*rand(M, K+2);  
    itemW=sqrt(1/K)*rand(N, K+2); 
    userW(:, 1)=1; 
    itemW(:, end)=1;
    LastError=0;
    item_sim = geo_alpha * spdiags(ones(N,1), 0, N, N) + (1 - geo_alpha) * item_sim;
    for e = 1 : max_iter
        tic;
        userW= APGUserLatentFactor(W, R, userW, item_sim*itemW, reg_u);        
        [itemW, Error]= APGItemLatentFactor(W, R, userW, itemW, reg_i, reg_s, item_sim, itemGroup);   
        CurrError = Error + 0.5* reg_u* norm(userW, 'fro')^2;
        deltaError=(CurrError - LastError)/abs(LastError);
        fprintf('Epoch %g, CurrError %g, LastError %g, DeltaErr %g, Time: %g\n', e, CurrError, LastError, deltaError, toc);
        if abs(deltaError) < tol            
            break;
        end
        LastError=CurrError;        
    end
    U = userW; V = item_sim * itemW;
    fprintf('complete the learning of the model parameters\n');
    %--------------- the sparsity of latent factors of item------------------------%
    fprintf('num_factors:%g, lambda:%g %g %g, ', K, reg_u, reg_i, reg_s);
    density=nnz(itemW)/numel(itemW);    
    fprintf('item latent factors density:%g\n', density);

end
