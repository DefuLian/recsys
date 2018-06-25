function metric = evaluate_rating(test, P, Q, k)
[I,J,V] = find(test);
pred_val = sum(P(I,:) .* Q(J,:), 2);
all_col = [I,J,V,pred_val];
rmse = sqrt(mean((V - pred_val).^2));
mae = mean(abs(V - pred_val));
act_col = sortrows(all_col, [1,-3]);
pred_col = sortrows(all_col,[1,-4]);
user_count = full(sum(test>0,2));
cum_user_count = cumsum(user_count);
cum_user_count = [0;cum_user_count];
num_users = size(test,1);
ndcg_all = zeros(num_users,k+1);
for u=1:num_users
    if user_count(u) > 0
        u_start = cum_user_count(u)+1;
        u_end = cum_user_count(u+1);
        act = act_col(u_start:u_end,3);
        n = user_count(u);
        discount = log2((1:n)'+1);
        pred = pred_col(u_start:u_end,3);
        %idcg = cumsum((2.^act - 1) ./ discount);
        idcg = cumsum(act ./ discount);
        %dcg = cumsum((2.^pred - 1) ./discount);
        dcg = cumsum(pred ./ discount);
        ndcg = dcg ./ idcg;
        if k > length(ndcg)
            ndcg = [ndcg; repmat(ndcg(end), k+1-length(ndcg),1)];
        else
            ndcg = [ndcg(1:k);ndcg(end)];
        end
        ndcg_all(u,:) = ndcg;
    end
end
ndcg_all = ndcg_all(user_count>0,:);
metric = struct('rating_ndcg', mean(ndcg_all(:,1:k)), 'rating_rmse', rmse, 'rating_mae', mae, 'rating_ndcgri', mean(ndcg_all(:,end)));
end
