function loss = compute_loss(Rt, Wt, P, Q, Xt, Y)
[~,M] = size(Rt);
Qt = Q.';
Pt = P.';
Yt = Y.';
X = Xt.';
loss = sum(sum(Rt .* Rt)) + sum(sum((Pt * P) .* (Qt * Q))) + sum(sum((Yt * Y) .* (Xt * X)));
loss = loss - 2*sum(sum(Y .* (Rt * X))) - 2*sum(sum(Q .* (Rt * P))) + 2*sum(sum((Yt *Q) .* (Xt * P)));
for i = 1:M
    x = Xt(:,i);
    r = Rt(:,i);
    w = Wt(:,i);
    Ind = w > 0;
    pred = r(Ind).' - P(i,:) * Qt(:,Ind) - x.' * Yt(:,Ind);
    loss = loss + sum(pred.^2 .* w(Ind).');
end
end