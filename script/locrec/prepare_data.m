dataset = 'Gowalla';
data = readContent(sprintf('/home/dlian/data/checkin/%s/data.txt', dataset));
[train, test] = split_matrix(data, 'i', 0.8);
sum(sum(train + test - data))
sum(sum(train>0 & test>0))
sum(sum(train>0)>0)
sum(sum(test>0)>0)
sum(sum(train>0)>0) + sum(sum(test>0)>0)

[I,J,V] = find(train);
dlmwrite(sprintf('/home/dlian/data/checkin/%s/train_influ.tsv',dataset),[I-1,J-1,V], 'delimiter', '\t', 'precision', '%d');
[I,J,V] = find(test);
dlmwrite(sprintf('/home/dlian/data/checkin/%s/test_influ.tsv',dataset),[I-1,J-1,V], 'delimiter', '\t', 'precision', '%d');
