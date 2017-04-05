mex -largeArrayDims iccf_sub.cpp
[train, test] = readData('D:\BaiduNetdiskDownload\checkin_rerun_tkde', 1);
metric1 = item_recommend(@iccf, R, 'test', test, 'topk', 200, 'max_iter', 10);
metric = item_recommend(@iccf, R, 'test', test, 'max_iter', 10);
metric = item_recommend(@iccf, R, 'split_ratio', 0.8, 'topk', 200, 'max_iter', 10, 'split_mode', 'en');
%output_wobias0 = iccf(R, 'K',50, 'max_iter', max_iter);

