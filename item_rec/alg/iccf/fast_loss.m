function loss = fast_loss(R, W, P, Q, a, d)
[M, N] = size(R);
if nargin == 4
    a = ones(M,1);
    d = ones(N,1);
elseif nargin == 5
    d = ones(N,1);
end

Rt = R.';

Qd = spdiags(d, 0, N, N) * Q;
Pa = spdiags(a, 0,M, M) * P;
QtQ = Q.' * Qd;
PtP = P.' * Pa;
loss = a.' * R.^2 *d + sum(sum((PtP) .* (QtQ))) - 2*sum(sum(Qd .* (Rt * Pa)));
Qt = Q.';
Wt = W.';
for i = 1:M
    r = Rt(:,i);
    w = Wt(:,i);
    Ind = w > 0;
    pred = r(Ind).' - P(i,:) * Qt(:,Ind);
    loss = loss + sum(pred.^2 .* w(Ind).');
end
end