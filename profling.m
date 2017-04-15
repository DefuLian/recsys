metric10 = item_recommend(@geomf, train, 'test', test, 'topk', 100, 'Y', item_grid, 'K', 50, 'reg_1', 10);
metric20 = item_recommend(@geomf, train, 'test', test, 'topk', 100, 'Y', item_grid, 'K', 50, 'reg_1', 20);
metric5 = item_recommend(@geomf, train, 'test', test, 'topk', 100, 'Y', item_grid, 'K', 50, 'reg_1', 5);


metric10_1 = item_recommend(@geomf, train, 'test', test, 'topk', 100, 'Y', item_grid, 'K', 50, 'reg_1', 10);


[U,V ] = geomf(train, 'Y', item_grid);
evaluate_item(train, test, [P], [Q], 100, 100);