function H_v = UpdateSVD(W)
%UpdateSVD: update rule in Eq.(16)
[b,n] = size(W);
m = mean(W,2);
JW = bsxfun(@minus,W,m);
JW = JW';
[P,ss] = eig(JW'*JW);
ss = diag(ss);
zeroidx = (ss <= 1e-10);
if sum(zeroidx) == 0
    H_v = sqrt(n)*P*(JW*P*diag(1./sqrt(ss)))';
else
    ss = ss(ss>1e-10);
    Q = JW*P(:,~zeroidx)*diag(1./sqrt(ss));
    Q = my_MGS(Q, b);
    H_v = sqrt(n)*P *Q';
end
% H_v = H_v';
end