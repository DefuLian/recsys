
[train, test] = readData('/home/dlian/data/subcheckin', 1);

data = train+test;
R = data>0;
metric1 = item_recommend(@iccf, train>0, 'test', test, 'topk', 200, 'max_iter', 10);
metric2 = item_recommend(@iccf, R, 'folds', 5, 'topk', 200, 'fold_mode', 'u');
%output_wobias0 = iccf(R, 'K',50, 'max_iter', max_iter);

