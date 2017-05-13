function [loss, Pred]= fast_loss(R, W, P, Q, a, d)
[M, N] = size(R);
if nargin == 4
    a = ones(M,1);
    d = ones(N,1);
elseif nargin == 5
    d = ones(N,1);
end

Rt = R.';

Qd = spdiags(d, 0, N, N) * Q;
Pa = spdiags(a, 0, M, M) * P;
QtQ = Q.' * Qd;
PtP = P.' * Pa;
loss0 = a.' * R.^2 *d + sum(sum((PtP) .* (QtQ))) - 2*sum(sum(Qd .* (Rt * Pa)));

[II,JJ,~] = find(W);
step = 10000;
num_step = floor((length(II) + step-1)/step);
pred = cell(num_step,1);
for i=1:num_step
    start_w = (i-1)*step +1;
    end_w = min(i * step, length(II));
    i_ind = II(start_w:end_w);
    j_ind = JJ(start_w:end_w);
    pred{i} = sum(P(i_ind,:) .* Q(j_ind,:), 2);
end
Pred = sparse(II,JJ,cell2mat(pred), M, N);
loss = loss0 + sum(sum(W .* (Pred - R).^2));
%Qt = Q.';
%Wt = W.';
%for i = 1:M
%    r = Rt(:,i);
%    w = Wt(:,i);
%    Ind = w > 0;
%    pred = r(Ind).' - P(i,:) * Qt(:,Ind);
%    loss = loss + sum(pred.^2 .* w(Ind).');
%end
end