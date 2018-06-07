function [ndcg,rmse] = rating_metric(test, P, Q, k)
[I,J,V] = find(test);
pred_val = sum(P(I,:) .* Q(J,:), 2);
all_col = [I,J,V,pred_val];
rmse = sqrt(mean((V - pred_val).^2));
act_col = sortrows(all_col, [1,-3]);
pred_col = sortrows(all_col,[1,-4]);
user_count = full(sum(test>0,2));
cum_user_count = cumsum(user_count);
cum_user_count = [0;cum_user_count];
num_users = size(test,1);
uind = 1;
ndcg_all = zeros(num_users - sum(sum(test>0)==0),k);
for u=1:num_users
    if user_count(u) == 0
        continue;
    end
    u_start = cum_user_count(u)+1;
    u_end = cum_user_count(u+1);
    act = act_col(u_start:u_end,3);
    discount = log2((1:k)'+1);
    pred = pred_col(u_start:u_end,3);
    if k > length(act)
        act_extend = [act; zeros(k-length(act),1)];
        pred_extend = [pred; zeros(k-length(act),1)];
    else
        act_extend = act(1:k);
        pred_extend = pred(1:k);
    end
    idcg = cumsum((2.^act_extend - 1) ./ discount);
    dcg = cumsum((2.^pred_extend - 1) ./discount);
    ndcg_all(uind,:) = dcg ./ idcg;
    uind = uind + 1;
end
ndcg = mean(ndcg_all);
end
