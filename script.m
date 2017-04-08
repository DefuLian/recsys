
[train, test] = readData('/home/dlian/data/subcheckin', 1);

data = train+test;
R = data>0;
metric1 = item_recommend(@iccf, train>0, 'test', test, 'topk', 200);
metric2 = item_recommend(@iccf, R,'split_ratio',0.8,'split_mode','en','topk', 200, 'times',3);
metric3 = item_recommend(@iccf, R, 'folds', 5, 'topk', 200, 'fold_mode', 'en');
[~, out] = item_recommend(@iccf, train>0, 'topk', 200);

%output_wobias0 = iccf(R, 'K',50, 'max_iter', max_iter);

