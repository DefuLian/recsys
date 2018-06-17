function s = ScaleScore(s,scale, maxS,minS)
%ScaleScore: scale the scores in user-item rating matrix to [-scale,
%+scale]. See footnote 2.
%s = s - mean(s);
%return
if maxS ~= minS
    s = (s-minS)/(maxS-minS);
    s = 2*scale*s-scale;
else
    s = s .* scale ./ maxS;
end
    
end