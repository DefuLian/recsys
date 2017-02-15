function [G,H,U,V,P,Q] = DCMFinit_fast(R,X,Y,r,varargin)

[maxItr, debug, alpha, lambda, beta, eta, test] = ...
   process_options(varargin, 'maxItr', 30, ...
                   'debug', true, 'alpha',0.01*[1,1],...
                   'lambda',[0, 0], 'beta',0, 'eta',[0.01 1], 'test', []);
gamma = (lambda+eta)+1e-10;
               
maxR = max(max(R));
minR = min(R(R>0));
RT = R.';
IDX = R~=0;
IDXT = IDX.';
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

[Ir, Jr, ~] = find(R);
eps = sparse(Ir, Jr, sum(G(Ir,:).* H(Jr,:),2), size(R,1), size(R,2));
err = R - eps;
err_t = err.';
eps_t = eps.';


converge = false;
it = 1;
disp('Starting DFMinit...');
while ~converge
    G0 = G;
    H0 = H;
    P0 = P;
    Q0 = Q;
    HT = H';
    HtH = HT * H;
    XU = X * U;
    for i = 1:m
        ind = IDXT(:,i);
        Hu = HT(:,ind);
        Si = nonzeros(RT(:,i));
        if isempty(Si)
            continue;
        end
        sub_err = err_t(ind, i);
        sub_eps = eps_t(ind, i);
        %Si = ScaleScore(Si,r,maxR,minR);
        %Si = (Si - mean(Si))./std(Si); Si = 1 ./ (1 + exp(-Si));
%         down = Hu*Hu'+beta*HtH+(alpha(1)+eta(1))*length(Si)*eye(r);
% dcmf(r_i, b_i, D_i, x_i, Ds, iter, beta, alpha, err, eps)
        G(i,:) = dcmf_init_mex(Si, G(i,:), Hu.', alpha(1) * P(i,:)'+eta(1)*XU(i,:)', HtH, 1, beta, alpha(1)+eta(1), sub_err);
        err_t(ind, i) = sub_err; 
    end
    err = err_t.';
    eps = eps_t.';
    GT = G';
    GtG = GT * G;
    YV = Y * V;
    for j = 1:n
        ind = IDX(:,j);
        Gi = GT(:,ind);
        Sj = nonzeros(R(:,j));
        if isempty(Sj)
            continue;
        end
        sub_err = err(ind, j);
        sub_eps = eps(ind, j);
        %Sj = ScaleScore(Sj,r,maxR,minR);
        %Sj = (Sj - mean(Sj))./std(Sj); Sj = 1 ./ (1 + exp(-Sj));
%         down = Gi*Gi'+beta*GtG+(alpha(2)+eta(2)) *length(Sj)* eye(r);
        H(j,:) = dcmf_init_mex(Sj, H(j,:), Gi.', alpha(2) * Q(j,:)' + eta(2) * YV(j,:)', GtG, 1, beta, alpha(2)+eta(2), sub_err);
        err(ind, j) = sub_err;
    end
    err_t = err.';
    eps_t = eps.';
    
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
        [ndcg_train,rmse_train] = rating_metric(R, G, H, 10);
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