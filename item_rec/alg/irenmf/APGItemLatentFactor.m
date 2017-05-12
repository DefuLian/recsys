function [itemW, Error]=APGItemLatentFactor(W, R, userW, itemW, reg_i, reg_s, item_sim, itemGroup)
	tau0 = 1e7; tau = tau0; tK0 = 1;  
    itemW0 = itemW;
    LastError = eps;
    init.userCorr = userW' * userW;
    init.subgX = (W .* R + R)' * userW;
    for e = 1 : 70
        tK= (1+ sqrt(1 + 4* tK0 * tK0))/2;
        X = itemW + ((tK0-1)/tK) * (itemW - itemW0); itemW0= itemW; tK0= tK;    
        [itemW, tau, error] = APGItemLFLineSearch(W, R, userW, X, init, tau, tau0, reg_i, reg_s, item_sim, itemGroup);         
        CurrError=error + 0.5*reg_i*norm(itemW, 'fro')^2 + ItemGroupLassoRegError(itemW, itemGroup, reg_s);
        deltaError=abs(CurrError - LastError)/abs(LastError);
        if deltaError < 1e-5   
            break;
        end
        LastError=CurrError;
    end
    Error=CurrError;
end
function [itemW, tau, error] = APGItemLFLineSearch(W, R, userW,  X, init, tau, tau0, reg_i, reg_s, item_sim, itemGroup)
    [M, N] = size(W);
    eta = 0.7;
    tau = eta * tau;
    itemW = item_sim * X ;
    [I, J, w] = find(W);
    a = sum(userW(I,:) .* itemW(J, :), 2) .* w;
    subgX = sparse(I, J, a, M, N)' * userW + itemW * init.userCorr - init.subgX; 
    subgX = item_sim' * subgX;
    norm2= norm(subgX, 'fro');
    pX= 0.5* fast_loss(R, W, userW, itemW);
    for e = 1 : 10
        Z = X- (1/tau) * subgX; 
        itemW = ItemProxyOperator(Z, itemGroup, reg_i, reg_s, tau); 
        norm1= norm(itemW - Z, 'fro');
        error = 0.5 * fast_loss(R, W, userW, item_sim * itemW);
        Qstau = 0.5 * tau * norm1^2 - 0.5/tau * norm2^2 + pX;
        if error <= Qstau
            break;
        else
            tau= min(tau/eta, tau0);
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
