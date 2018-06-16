function [G,H,U,V,P,Q] = DCMFinit(R,X,Y,r,varargin)

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


converge = false;
it = 1;
disp('Starting DFMinit...');
while ~converge
    G0 = G;
    H0 = H;
    P0 = P;
    Q0 = Q;
    HT = H';
    if beta > 0
        HtH = HT * H;
    end
    XU = X * U;
    for i = 1:m
        Hu = HT(:,IDXT(:,i));
        Si = nonzeros(RT(:,i));
        if isempty(Si)
            continue;
        end
        %Si = ScaleScore(Si,r,maxR,minR);
        %Si = (Si - mean(Si))./std(Si); Si = 1 ./ (1 + exp(-Si));
%         down = Hu*Hu'+beta*HtH+(alpha(1)+eta(1))*length(Si)*eye(r);
        down = Hu*Hu'+(alpha(1)+eta(1))*eye(r);
        if beta>0
            down = down + beta*HtH;
        end
        up = Hu*Si + alpha(1) * P(i,:)'+eta(1)*XU(i,:)';
        g = down\up;
        G(i,:) = g';
    end
    GT = G';
    if beta > 0
        GtG = GT * G;
    end
    YV = Y * V;
    for j = 1:n
        Gi = GT(:,IDX(:,j));
        Sj = nonzeros(R(:,j));
        if isempty(Sj)
            continue;
        end
        %Sj = ScaleScore(Sj,r,maxR,minR);
        %Sj = (Sj - mean(Sj))./std(Sj); Sj = 1 ./ (1 + exp(-Sj));
%         down = Gi*Gi'+beta*GtG+(alpha(2)+eta(2)) *length(Sj)* eye(r);
        down = Gi*Gi'+(alpha(2)+eta(2))*eye(r);
        if beta >0
            down = down + beta*GtG;
        end
        up = Gi*Sj + alpha(2) * Q(j,:)' + eta(2) * YV(j,:)';% linear term
        h = down\up;
        H(j,:) = h';
    end
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