[train, test] = readData('/home/dlian/data/test/', 0);
load('/home/dlian/data/test/BerlinSubCheckins_random_0.7_50_clusters.mat')
load('/home/dlian/data/test/cor.mat')

[M, N] = size(train);
group_num = max(clusterInx(:,2));
item_group = sparse(clusterInx(:,1), clusterInx(:,2), true, N, group_num);

%[U,V] = irenmf(train, 'alpha', 10, 'K', 200, 'item_sim', item_sim, 'itemGroup', item_group);
metric = item_recommend(@(mat) irenmf(mat, 'alpha', 10, 'K', 200, 'item_sim', item_sim, 'itemGroup', item_group, 'max_iter',50), train, 'test', test);