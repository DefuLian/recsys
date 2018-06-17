function [G,H,U,V,P,Q] = DCMFinit_fast(R,X,Y,r,varargin)

[maxItr, debug, alpha, lambda, beta, eta, test, is_classifier] = ...
   process_options(varargin, 'maxItr', 30, ...
                   'debug', true, 'alpha',0.01*[1,1],...
                   'lambda',[0, 0], 'beta',0, 'eta',[0.01 1], 'test', [], 'is_classifier',false);
               
gamma = (lambda+eta)+1e-10;
               
RT = R.';

f = size(X, 2);
l = size(Y, 2);
rng(10)
[m,n] = size(R);
G =  (randn(r,m) * 0.01)';
H =  (randn(r,n) * 0.01)';
P = UpdateSVD(G')';
Q = UpdateSVD(H')';
U = (X'*X*(lambda(1)+eta(1)) + gamma(1)*speye(f))\(X'*(lambda(1)*P+eta(1)*G));
V = (Y'*Y*(lambda(2)+eta(2)) + gamma(2)*speye(l))\(Y'*(lambda(2)*Q+eta(2)*H));
tol = 1e-5;


converge = false;
it = 1;
disp('Starting DFMinit...');
while ~converge
    G0 = G;
    H0 = H;
    P0 = P;
    Q0 = Q;
    XX = alpha(1) * P + eta(1)* X * U;
    if beta>0
        Hs = H.'* H * beta;
    else
        Hs = [];
    end
    G = dcmf_init_all_mex(RT, H, G, XX, Hs, 1, alpha(1)+eta(1), is_classifier);
    
    YY = alpha(2) * Q + eta(2) * Y * V;
    if beta>0
        Gs = G.'* G * beta;
    else
        Gs = [];
    end
    H = dcmf_init_all_mex(R, G, H, YY, Gs, 1, alpha(2)+eta(2), is_classifier);
    
    P = UpdateSVD((G + lambda(1)/alpha(1)*X*U)')';
    Q = UpdateSVD((H + lambda(2)/alpha(2)*Y*V)')';
%     U = (X'*X + gamma(1)/lambda(1)*speye(f))\(X'*P);
%     V = (Y'*Y + gamma(2)/lambda(2)*speye(l))\(Y'*Q);
    U = (X'*X*(lambda(1)+eta(1)) + gamma(1)*speye(f))\(X'*(lambda(1)*P+eta(1)*G));
    V = (Y'*Y*(lambda(2)+eta(2)) + gamma(2)*speye(l))\(Y'*(lambda(2)*Q+eta(2)*H));
    
    if it >= maxItr || max([norm(G-G0,'fro') norm(H-H0,'fro') norm(P-P0,'fro') norm(Q-Q0,'fro')]) < max([m n])*tol
        converge = true;
    end
    
    fprintf('DFMinit Iteration:%d, ', it);
    if debug
        %[obj,rmse] = DCMFobj(R,G,H,P,Q,U,V,X,Y,alpha,lambda,beta,gamma,eta);
        [ndcg_train,rmse_train] = rating_metric(train, G, H, 10);
        fprintf('rmse_train=%.5f, ndcg_train =%.5f', rmse_train, ndcg_train(10));
        if ~isempty(test)
            [ndcg_test,rmse_test] = rating_metric(test, G, H, 10);
            fprintf(', rmse_test=%.5f, ndcg_test=%.5f', rmse_test, ndcg_test(10));
        end
        fprintf('\n');
    end
    it = it+1;
end
end