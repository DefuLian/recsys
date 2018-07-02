function mat = explicit2implicit(data)
avg_score = sum(data,2)./sum(data~=0,2); 
%std_score = sqrt(sum(test.^2, 2) ./ sum(test~=0,2) - avg_score); %avg_score = avg_score + 2*std_score;
avg_score = min(avg_score, max(data,[],2)-1e-3);
m = size(data,1); avg_matrix = spdiags(avg_score, 0, m, m) * (data~=0) ;
mat = +(data>avg_matrix);
end