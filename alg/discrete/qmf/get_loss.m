load('C:\Users\USTC\Desktop\data\ml10Mdata.mat');
[train, test] = split_matrix(data, 'un', 0.8);
[P_qcf_rand, Q_qcf_rand] = qcf(train, 'max_iter', 20, 'K', 64, 'alpha', 5, 'rand_init', true, 'test', test);
[P,Q] = iccf(train, 'max_iter', 20, 'K', 64, 'alpha', 5);
[P_qcf, Q_qcf] = qcf(train, 'max_iter', 20, 'K', 64, 'alpha', 5, 'rand_init', false, 'test', test);
[B,D] = dmf(train, 'max_iter', 20, 'K', 64, 'rho', 1.0/5, 'alpha', 0, 'beta', 0);
save('C:\Users\USTC\Desktop\data\ml10M_para.mat', 'Q', 'Q_qcf_rand', 'Q_qcf', 'D')