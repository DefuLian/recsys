function W = proj_stiefel_manifold(A)
%%% min_W |A - W|_F^2, s.t. W^T W = I
[U, ~, V] = svd(A, 0);
W = U * V.';
end