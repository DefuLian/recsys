function [B, D, B1, D1, P, Q, U, V] = DCMF_fast(train, X, Y, r, varargin)

% C -- user-location frequency matrix. (m,n)
% P -- latent factors of users. (m,l)
% Q -- latent factors of locations. (n,l)
% B -- binary codes of P. (m,r)
% D -- binary codes of Q. (n,r)
% X -- features of users. (m,f)
% Y -- features of locations. (n,f)
% U -- latent factors of features of users. (f,l)
% V -- latent factors of features of locations. (f,l)
% l -- the number of latent factors
% f -- the number of features
% r -- bit length (r==l)
% lambda -- regularized parameters
% alpha -- trade-off paramter. good default = 0.001.
% beta -- trade-off paramter. good default = 0.001.
% gama -- regularized parameters for U,V subproblem.
% option :
%     .B0,D0,P0,Q0 -- 
%     .maxItr -- maximun iteration times
%     .debug -- check loss function
% Written by Rui Liu
% initialization



f = size(X, 2);
l = size(Y, 2);

[maxItr, maxItr2, debug, B0, D0, P0, Q0, U0, V0, alpha, lambda, beta, eta, test, is_classifier] = ...
   process_options(varargin, 'maxItr', 30, 'maxItr2', 5, ...
                   'debug', true, 'B0', [], 'D0',[], ...
                   'P0', [], 'Q0', [], 'U0', [], 'V0',[], 'alpha',[0.01,0.01],...
                   'lambda',[0, 0], 'beta',0, 'eta', [0 0], 'test', [], 'is_classifier', false);

if ~is_classifier
    R = scale_matrix(train, r);
end
RT = R.';
gamma = (lambda+eta) + 1e-10;
               
if isempty(B0) || isempty(D0) ||isempty(P0) ||isempty(Q0) ||(size(U0,1)>0 && isempty(U0)) || (size(V0,1)>0 && isempty(V0))
    [G,H,U0,V0,P0,Q0] = DCMFinit(R, X, Y, r, 'maxItr',maxItr, 'alpha', alpha, 'lambda', lambda, 'beta', beta, 'eta', eta, 'debug', debug);
    B0 = sign(G); B0(B0 == 0) = 1;
    D0 = sign(H); D0(D0 == 0) = 1;
end

% debug = 0;
converge = false;
it = 1;


B1 = B0; B = B0+0; 
D1 = D0; D = D0+0;
P = P0;
Q = Q0;
U = U0;
V = V0;
% Update iteratively
while ~converge
    BB = B+0; DD = D+0;
    XX = alpha(1) * P + eta(1)* X * U;
    if beta >0
        Ds = D.'* D * beta;
    else
        Ds = [];
    end
    B = dcmf_all_mex(RT, D, B, XX, Ds, maxItr2, is_classifier);
    
    YY = alpha(2) * Q + eta(2) * Y * V;
    if beta>0
        Bs = B.'*B * beta;
    else
        Bs = [];
    end
    D = dcmf_all_mex(R, B, D, YY, Bs, maxItr2, is_classifier);
    
    P = UpdateSVD((B + lambda(1)/alpha(1)*X*U)')';
    Q = UpdateSVD((D + lambda(2)/alpha(2)*Y*V)')';
%     U = (X'*X + gamma(1)/(lambda(1)+eta(1))*speye(f))\(X'*(lambda(1)*P+eta(1)*B)/(lambda(1)+eta(1)));
%     V = (Y'*Y + gamma(2)/(lambda(2)+eta(2))*speye(l))\(Y'*(lambda(2)*Q+eta(2)*D)/(lambda(2)+eta(2)));
    
    U = ((lambda(1)+eta(1))*(X'*X) + gamma(1)*speye(f))\(X'*(lambda(1)*P+eta(1)*B));
    V = ((lambda(2)+eta(2))*(Y'*Y) + gamma(2)*speye(l))\(Y'*(lambda(2)*Q+eta(2)*D));
    
    
    fprintf('Iteration times:%d, ', it);

    if debug
        [ndcg_train,rmse_train] = rating_metric(R, B, D, 10);
        fprintf('rmse_train=%.5f, ndcg_train =%.5f', rmse_train, ndcg_train(10));
        if ~isempty(test)
            [ndcg_test,rmse_test] = rating_metric(test, B, D, 10);
            fprintf(', rmse_test=%.5f, ndcg_test=%.5f', rmse_test, ndcg_test(10));
        end
        fprintf('\n');
        %[obj0, loss] = DCMFobj(R,B,D,P,Q,U,V,X,Y,alpha,lambda,beta,gamma,eta);
        %disp(['obj value = ',num2str(obj0)]);
        %disp(['rmse value = ',num2str(loss)])
    end

    
    % Judge whether converge
    if it >= maxItr || (sum(sum(B~=BB)) == 0 && sum(sum(D~=DD)) == 0)
        converge = true;
    end
    it = it + 1;
    
end
end

function R = scale_matrix(R, s)
%ScaleScore: scale the scores in user-item rating matrix to [-scale,
%+scale]. See footnote 2.
%s = s - mean(s);
%return
maxS = max(max(R));
minS = min(R(R>0));
[I, J, V] = find(R);
if maxS ~= minS
    VV = (V-minS)/(maxS-minS);
    VV = 2 * s * VV - s + 1e-10;
else
    VV = V .* s ./ maxS;
end
R = sparse(I, J, VV, size(R,1), size(R,2));
end

% function b = Updatebit(b,E,d,AP,Beta,maxItr2)
% r = length(b);
% converge = 0;
% it = 1;
% while(~converge)
%     count = 0;
%     for k = 1:r
%         bb = d(k,:) * (E+(b(k)*d(k,:))') + AP(k) - Beta(k);
%         if bb > 0
%             if b(k) == 1
%                 count = count + 1;
%             else
%                 b(k) = 1;
%             end
%         elseif bb < 0
%             if b(k) == -1
%                 count = count + 1;
%             else
%                 b(k) = -1;
%             end
%         else
%             count = count + 1;
%         end
%     end
%     if count == r || it >= maxItr2
%         converge = 1;
%     end
%     it = it + 1;
% end
% end