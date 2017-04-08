function [ X ] = WNNLS( R, Y, varargin )
%WNNLS Weighted Non-negative least square
%   Detailed explanation goes here
[M, ~] = size(R);
[epsilon, reg, X] = process_options(varargin, 'epsilon', 1e-10, 'reg', 0,...
    'X', sparse(M, size(Y,2)));
W = GetWeight(R, 'epsilon', epsilon);
user_cell = cell(M,1);
item_cell = cell(M,1);
val_cell = cell(M,1);
Yt = Y.';
YtY = Yt * Y;
Xt = X.';
Wt = W.';
R = R > 0;
Rt = R.';
parfor u = 1:M
    fprintf('%d\n', u);
    x = Xt(:, u);
    w = Wt(:, u);
    r = Rt(:, u);
    Ind = w>0 | r>0; wu = w(Ind); Wu = spdiags(wu, 0, length(wu), length(wu));
    ru = r(Ind);
    sub_Y = Y(Ind, :);
    sub_Yt = Yt(:, Ind);
    YCY = sub_Yt * Wu * sub_Y + YtY;
    grad_invariant = - sub_Yt * (wu .* ru + ru) + reg;
    x = LineSearch(YCY, grad_invariant, x);
    [loc, I, val ] = find(x);
    ind = val > 1e-6;
    user_cell{u} = u * I(ind);
    item_cell{u} = loc(ind);
    val_cell{u} = val(ind);
    %X = X + sparse(u*I, loc, val, M, len(x));
    %gradX = gradX + projnorm ^2;
end
X = sparse(cell2mat(user_cell), cell2mat(item_cell), cell2mat(val_cell), size(X, 1), size(X, 2));
end
