function [U,V,X,Y] = DCFinit(S, r, alpha, beta, option)
%DCFinit: Initialization for Dicrete Collaborative Filtering as in Eq.(17)

%Input:
%maxS: max rating score
%minS: min rating score
%S: user-item score matrix, [m,n] = size(S)
%ST: transpose of ST, for efficient sparse matrix indexing in Matlab, i.e.,
%matlab can only efficiently access sparse matrix by column.
%IDX: nonzero (observed) entry index of S
%IDXT: transpose of IDX for efficient sparse matrix indexing in Matlab.
%r: bit length
%alpha: trade-off paramter. good default = 0.001.
%beta: trade-off paramter. good default = 0.001.
%option:
%option.maxItr: max iterations. Default = 50.
%option.tol: tolerance. Default = 1e-5.
%option.debug: show obj?. Default = false.

%Output:
%U: user vector
%V: item vector
%X: surrogate user vector
%Y: surrogate item vector

%Reference:
%   Hanwang Zhang, Fumin Shen, Wei Liu, Xiangnan He, Huanbo Luan, Tat-seng
%   Chua. "Discrete Collaborative Filtering", SIGIR 2016

%Version: 1.0
%Written by Hanwang Zhang (hanwangzhang AT gmail.com)

ST = S';
IDX = (S~=0);
IDXT = IDX';
maxS = max(max(S));
minS = min(S(S>0));
[m,n] = size(S);
rng(10)
U =  randn(r,m) * 0.01;
V =  randn(r,n) * 0.01;
% U = rand(r,m);
% V = rand(r,n);
X = UpdateSVD(U);
Y = UpdateSVD(V);
if isfield(option,'maxItr')
    maxItr = option.maxItr;
else
    maxItr = 20;
end
if isfield(option,'tol')
    tol = option.tol;
else
    tol = 1e-5;
end
if isfield(option,'debug')
    debug = option.debug;
else
    debug = false;
end
converge = false;
it = 1;


if debug
    disp('Starting DCFinit...');
    disp(['obj value = ',num2str(DCFinitObj(maxS, minS,S,ST,IDX,U,V,X,Y,alpha,beta))]);
end

while ~converge
    U0 = U;
    V0 = V;
    X0 = X;
    Y0 = Y;
    for i = 1:m
        Vi = V(:,IDXT(:,i));
        Si = nonzeros(ST(:,i));
        if isempty(Si)
            continue;
        end
        Si = ScaleScore(Si,r,maxS,minS);
%         Q = Vi*Vi'+alpha*length(Si)*eye(r);
%         L = Vi*Si+2*alpha*X(:,i);
        Q = Vi*Vi'+alpha*eye(r);
        L = Vi*Si+alpha*X(:,i);
        U(:,i) = Q\L;
    end
    for j = 1:n
        Uj = U(:,IDX(:,j));
        Sj = nonzeros(S(:,j));
        if isempty(Sj)
            continue;
        end
        Sj = ScaleScore(Sj,r,maxS,minS);
%         Q = Uj*Uj'+beta*length(Sj)*eye(r);%quadratic term
%         L = Uj*Sj+2*beta*Y(:,j);% linear term
        Q = Uj*Uj'+beta*eye(r);%quadratic term
        L = Uj*Sj+beta*Y(:,j);% linear term
        V(:,j) = Q\L;
    end
    
    
    X = UpdateSVD(U);
    Y = UpdateSVD(V);
    
    disp(['DCFinit Iteration:',int2str(it-1)]);
    if it >= maxItr || max([norm(U-U0,'fro') norm(V-V0,'fro') norm(X-X0,'fro') norm(Y-Y0,'fro')]) < max([m n])*tol
        converge = true;
    end
    
    if debug
        disp(['obj value = ',num2str(DCFinitObj(maxS, minS,S,ST,IDX,U,V,X,Y,alpha,beta))]);
    end
    it = it+1;
end
end


function obj = DCFinitObj(maxS, minS, S, ST,IDX,B, D, X, Y, alpha, beta)
[m,n] = size(S);
r = size(B,1);
loss = zeros(1,n);
for j = 1:n
    dj = D(:,j);
    Bj = B(:,IDX(:,j));
    BBj = Bj*Bj';
    term1 = dj'*BBj*dj;
    Sj = ScaleScore(nonzeros(S(:,j)),r,maxS,minS);
    term2 = 2*dj'*Bj*Sj;
    term3 = sum(Sj.^2);
    loss(j) = term1-term2+term3;
end
loss = sum(loss);

reg = 0;
for i = 1:m
    bi = B(:,i);
    Si = nonzeros(ST(:,i));
    reg = reg+alpha*length(Si)*sum(bi.^2);
end

for j = 1:n
    dj = D(:,j);
    Sj = nonzeros(S(:,j));
    reg = reg+beta*length(Sj)*sum(dj.^2);
end

obj = loss+reg-2*alpha*trace(B*X')-2*beta*trace(D*Y');
end
