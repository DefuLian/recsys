[train, test] = readData('C:\Users\134162\Downloads\IRenMF\datasets', 0);
load('C:\Users\134162\Downloads\IRenMF\datasets\BerlinSubCheckins_random_0.7_50_clusters.mat')
load('C:\Users\134162\Downloads\IRenMF\datasets\cor.mat')

[M, N] = size(train);
group_num = max(clusterInx(:,2));
item_group = sparse(clusterInx(:,1), clusterInx(:,2), true, N, group_num);

[U,V] = irenmf(train, 'alpha', 10, 'K', 200, 'item_sim', item_sim, 'itemGroup', item_group);