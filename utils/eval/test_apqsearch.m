load C:/Users/USTC/Desktop/gowalladata.mat
[train, test] = split_matrix(data, 'un', 0.8);
[P, Q]=iccf(train, 'alpha', 20, 'K', 64);
idx = randi(size(P,1), 1000, 1);
U=P(idx,1:64);
V=Q(:,1:64);
train1 = train(idx,:);
test1 = test(idx,:);
[code0, center0] = kmeans(V, 256);
V1 = center0(code0,:);
[code1, center1] = kmeans(V-V1, 256);
V2 = center1(code1,:);
VV = V1 + V2; 

code = uint32([code0,code1]); 
center = [center0, center1];
result = apq_search(U, code, center, train', 20);
result1 = topk_lookup(U, VV, train', 20, true);
sum(sum(abs(result-result1)))

m = mip_search(struct('real', U, 'query', U), struct('real', VV, 'code', code, 'word', center), train1, test1, 100);
m.item_recall(1,100)
m = mip_search(U, VV, train1, test1, 100);
m.item_recall(1,100)

[code0, center0] = kmeans(V(:,1:32),256);
[code1, center1] = kmeans(V(:,33:64),256);
V1 = center0(code0,:);
V2 = center1(code1,:);
VV = [V1,V2];
code = uint32([code0,code1]); 
center = [center0, center1];

result = apq_search(U, code, center, train', 20);
result1 = topk_lookup(U, VV, train', 20, true);
sum(sum(abs(result-result1)))

m = mip_search(struct('real', U, 'query', U), struct('real', VV, 'code', code, 'word', center), train1, test1, 100);
m.item_recall(1,100)
m = mip_search(U, VV, train1, test1, 100);
m.item_recall(1,100)