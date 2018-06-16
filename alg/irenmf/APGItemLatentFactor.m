function itemW=APGItemLatentFactor(W, R, userW, itemW, reg_i, reg_s, item_sim, itemGroup)
	tau = 1e7; tK0 = 1;  
    itemW0 = itemW;
    LastError = eps;
    init.userCorr = userW' * userW;
    init.subgX = (W .* R + R)' * userW;
    for e = 1 : (150*2)
        tK= (1+ sqrt(1 + 4* tK0 * tK0))/2;
        X = itemW + ((tK0-1)/tK) * (itemW - itemW0); itemW0= itemW; tK0= tK;    
        [itemW, tau, CurrError] = APGItemLFLineSearch(W, R, userW, X, init, tau, reg_i, reg_s, item_sim, itemGroup);         
        %CurrError=error + 0.5*reg_i*norm(itemW, 'fro')^2 + ItemGroupLassoRegError(itemW, itemGroup, reg_s);
        deltaError=(CurrError - LastError)/abs(LastError);
        if e>1 && (deltaError>0 || abs(deltaError) < 1e-5)
            break;
        end
        LastError=CurrError;
    end
end
function [itemW, tau, error] = APGItemLFLineSearch(W, R, userW,  X, init, tau, reg_i, reg_s, item_sim, itemGroup)
    eta = 0.5;
    itemW = item_sim * X ;
    [pX, pred]= fast_loss(R, W, userW, itemW);
    subgX = (pred .* W)' * userW + itemW * init.userCorr - init.subgX; 
    subgX = item_sim' * subgX;
    norm2= norm(subgX, 'fro');
    for e = 1 : 50
        Z = X- (1/tau) * subgX; 
        itemW=(tau/(tau + reg_i))*Z;
        %itemW = ItemProxyOperator(Z, itemGroup, reg_i, reg_s, tau); 
        norm1= norm(itemW - Z, 'fro');
        error = 0.5 * fast_loss(R, W, userW, item_sim * itemW);
        Qstau = 0.5 * tau * norm1^2 - 0.5/tau * norm2^2 + 0.5 * pX;
        suff_decr = error <= Qstau;
        if e == 1
            decr_tau = suff_decr; itemW_old = X;
        end
        if decr_tau
            if ~suff_decr || norm(itemW_old-itemW, 2)<eps
                if ~suff_decr
                    tau = tau/eta;
                end
                itemW = itemW_old; break
            else
                tau = tau * eta; itemW_old = itemW;
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

function Stau=ItemProxyOperator(Z, itemGroup, reg_i, reg_s, tau)
    [N, K]= size(Z);  
    Stau=zeros(N, K);    
    group_num=size(itemGroup, 2);
    for d =1 : K         
        for g= 1 : group_num
            %subInX=InX(InXStat(1, g) : InXStat(2, g), 1);
            ind = itemGroup(:, g)>0;
            Zgd=Z(ind, d);  omega=sqrt(nnz(ind));
            l2norm=norm(Zgd, 2);  theta= reg_s*omega/tau; 
            if l2norm > theta
                Stau(ind, d)= ((l2norm - theta)/(l2norm * (1 + reg_i/tau)))*Zgd;
            else
                Stau(ind, d)= zeros(length(Zgd), 1);
            end              

        end
    end
end

function regError=ItemGroupLassoRegError(itemW, itemGroup, reg_s)
    group_num=size(itemGroup, 2);
    K = size(itemW, 2); regError=0;
    for d= 1 : K
        for g =1 : group_num 
            ind=itemGroup(:,g)>0;
            subvec=itemW(ind, d); omega=sqrt(nnz(ind));
            regError= regError + omega* norm(subvec, 2);
        end
    end
    regError= regError * reg_s;
end
