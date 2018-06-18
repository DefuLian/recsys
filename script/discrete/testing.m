dir = 'C:/Users/liand/Desktop/code/dataset';

K = 64; max_iter = 20;

dataset = 'ml10Mdata';
load(sprintf('%s/%s.mat', dir , dataset))
Traindata(Testdata>0)=0;
data = Traindata + Testdata;

pph_ = @(mat,varargin) pph(mat, 'max_iter', max_iter, 'K', K, varargin{:});
[metric_pph_ml10m, metric_pph_detail_ml10m] = running(pph_, data, 'lambda', [0.01,0.5*[1,2,4,8,16,32]]);
[metric_pph_ml10m, metric_pph_detail_ml10m] = running(pph_, data, 'lambda', 0.1);


dataset = 'yelpdata';
load(sprintf('%s/%s.mat', dir , dataset))
Traindata(Testdata>0)=0;
data = Traindata + Testdata;

[metric.item, metric_detail.item] = item_recommend(@(mat) pph_(mat, best_para{:}), dataset, 'test_ratio', 0.2, 'times', 5);

