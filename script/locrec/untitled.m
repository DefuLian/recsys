a = [metric_geo_cv, metric_K_graph{3}, metric_irenmf];
recall = cell2mat({a.recall}');
recall = recall(1:2:end,5:5:200);

[a,b]=split_matrix(test, 'u', 0.33);
[I,J,V] = find(a);
dlmwrite('/home/dlian/data/checkin/Beijing/subtest.txt', [I-1,J-1,V], 'delimiter', '\t', 'precision', '%d');