function userW = APGUserLatentFactor(W, R, userW, itemW, reg_u)
    tau0 = 1e6; tau = tau0; tK0=1;  
    userW0= userW;
    LastError=eps;
    init.itemCorr = itemW'*itemW;
    init.subgX= (W .* R + R) * itemW;    
    for e = 1 : 70
        tK = (1+ sqrt(1 + 4*tK0*tK0))/2;
        X = userW + ((tK0-1)/tK) * (userW - userW0); tK0= tK;  userW0= userW; 
        [userW, tau, error] = APGUserLFLineSearch(W, R, X, itemW, init, tau, tau0, reg_u);        
        CurrError=error + 0.5* reg_u * norm(userW, 'fro')^2;
        deltaError=abs(CurrError - LastError)/abs(LastError);
        if deltaError < 1e-5          
            break;
        end
        LastError=CurrError;
    end
end
function [userW, tau, error] = APGUserLFLineSearch(W, R, X, itemW, init, tau, tau0, reg_u)
    [M, N] = size(W);
    eta = 0.7;
    tau = eta * tau;
    [I, J, w] = find(W);
    a = sum(X(I,:) .* itemW(J, :), 2) .* w;
    subgX = sparse(I, J, a, M, N) * itemW + X*init.itemCorr - init.subgX; 
    pX= 0.5* fast_loss(R, W, X, itemW);
    norm2= norm(subgX, 'fro');    
    for e = 1 : 10
        Z= X - (1/tau)*subgX; 
        userW=(tau/(tau + reg_u))*Z;
        error = 0.5 * fast_loss(R, W, userW, itemW);
        norm1= norm(userW - Z, 'fro');
        Qstau = 0.5 * tau * norm1^2 - 0.5/tau * norm2^2 + pX;
        if  error<=Qstau           
            break;
        else
            tau= min(tau/eta, tau0);
        end
    end
end

