function [obj,loss] = DCMFobj(R, B, D, P, Q, U, V, X, Y, alpha,lambda,beta,gamma, eta)
n = size(R, 2);
maxR = max(max(R));
minR = min(R(R>0));
IDX = R~=0;

f = size(X, 2);
l = size(Y, 2);
r = size(B, 2);

loss = 0;
% reg1 = zeros(1,m);
% reg2 = zeros(1,n);
BT = B';
if beta>0
    BtB = beta * BT * B;
end
for j = 1:n
    b = BT(:,IDX(:,j));
    d = D(j,:);
    r0 = ScaleScore(nonzeros(R(:,j)),r,maxR,minR);
    bb = b*b';
    term1 = d*bb*d';
    term2 = 2*d*b*r0;
    term3 = sum(r0.^2);
    loss = loss + term1 - term2 + term3;
    if beta >0
        loss = loss + d * BtB * d';
    end
end
loss = loss + alpha(1) * sum(sum(B.^2)) + alpha(2)*sum(sum(D.^2));
obj = loss + (alpha(1) + lambda(1)) * sum(sum(P.^2)) + (alpha(2) + lambda(2))* sum(sum(Q.^2));
obj = obj - 2*trace(P'*(alpha(1)*B+lambda(1)*X*U)) ;
obj = obj - 2*trace(Q'*(alpha(2)*D+lambda(2)*Y*V));
obj = obj + trace(U'*(lambda(1)*(X'*X)+gamma(1)*speye(f))*U);
obj = obj + trace(V'*(lambda(2)*(Y'*Y)+gamma(2)*speye(l))*V);
obj = obj + eta(1)* norm(B-X*U,'fro').^2 + eta(2) * norm(D - Y*V, 'fro').^2;
loss = sqrt(loss/nnz(R));
end
