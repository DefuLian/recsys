function userW = APGUserLatentFactor(W, R, userW, itemW, reg_u)
    tau = 1e6; tK0=1;  
    userW0= userW;
    LastError=eps;
    init.itemCorr = itemW'*itemW;
    init.subgX= (W .* R + R) * itemW;    
    for e = 1 : 70
        tK = (1+ sqrt(1 + 4*tK0*tK0))/2;
        X = userW + ((tK0-1)/tK) * (userW - userW0); tK0= tK;  userW0= userW; 
        [userW, tau, error] = APGUserLFLineSearch(W, R, X, itemW, init, tau, reg_u);        
        CurrError=error + 0.5* reg_u * norm(userW, 'fro')^2;
        deltaError=(CurrError - LastError)/abs(LastError);
        if e>1 && (deltaError>0 || abs(deltaError) < 1e-5)
            break;
        end
        LastError=CurrError;
    end
end
function [userW, tau, error] = APGUserLFLineSearch(W, R, X, itemW, init, tau, reg_u)
    eta = 0.5;
    [pX, pred]= fast_loss(R, W, X, itemW);
    subgX = (pred .* W) * itemW + X*init.itemCorr - init.subgX; 
    norm2= norm(subgX, 'fro');   
    for e = 1 : 50
        Z= X - (1/tau)*subgX; 
        userW=(tau/(tau + reg_u))*Z;
        error = 0.5 * fast_loss(R, W, userW, itemW);
        norm1= norm(userW - Z, 'fro');
        Qstau = 0.5 * tau * norm1^2 - 0.5/tau * norm2^2 + 0.5 * pX;
        suff_decr = error <= Qstau;
        if e == 1
            decr_tau = suff_decr; userW_old = X;
        end
        if decr_tau
            if ~suff_decr || norm(userW_old-userW, 2)<eps
                if ~suff_decr
                    tau = tau/eta;
                end
                userW = userW_old; break
            else
                tau = tau * eta; userW_old = userW;
            end
        else
            if suff_decr
                break;
            else
                tau = tau/eta; 
            end
        end
    end
end

