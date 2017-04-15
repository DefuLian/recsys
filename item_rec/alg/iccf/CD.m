function U = CD(P, U, X, reg, tol)
% coordinate descent for optimizing L = || P - X U ||_F^2 + reg ||U||_F^2, 
% P: MxK matrix, X: MxL matrix, U: LxK matrix

tol = min(tol, 1e-3);
F = size(X, 2);
prev_loss = norm(P - X * U, 'fro')^2 + reg * norm(U, 'fro')^2;
X2 = sum(X.^2).';
for iter=1:1000
    err = P - X * U;
    for f=1:F
        xf = X(:,f);
        uf = U(f,:);
        U(f,:) = (xf.' * err + uf * X2(f))./(reg+X2(f));
        err = err + xf * (uf - U(f,:));
    end
    loss = norm(P - X * U, 'fro')^2 + reg * norm(U, 'fro')^2;
    fprintf('loss=%f\n',loss);
    if prev_loss > loss && (prev_loss - loss)/ prev_loss < tol
        break
    end
    prev_loss = loss;
end
end