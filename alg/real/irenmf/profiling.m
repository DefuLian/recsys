[M, N] = size(train);
group_num = max(clusterInx(:,2));
item_group = sparse(clusterInx(:,1), clusterInx(:,2), true, N, group_num);
[U,V] = irenmf(train, 'alpha', 10, 'K', 200, 'item_sim', item_sim, 'itemGroup', item_group);