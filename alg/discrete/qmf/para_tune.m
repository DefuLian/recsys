run('~/software/recsys/setup.m')
parpool('local', 13);
dir = '~';
for dataset = {'amazon', 'yelp'}
    fprintf('%s\n', dataset{1});
    dataset_path = sprintf('%s/data/%sdata.mat', dir, dataset{1});
    result_dir = sprintf('%s/result_qcf/', dir);
    qcf_dim_tune(dataset_path, result_dir, false)
end