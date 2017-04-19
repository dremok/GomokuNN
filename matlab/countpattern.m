function c = countpattern(b, p, f)

% b: board
% p: player 
% f: features

c = zeros(f.group(end),1);
for j = 1:size(f.pattern,1)
    c(f.group(j)) = c(f.group(j)) + countpat(b*p,f.pattern{j});
end

%% Test it:
% board = [0  0  1  1  0  0  0  0  0  0  0  0  0
%          0  1  0  0  0  0  0  0  0  0  0  0  0
%          0  0  0  0  0  0  0  0  0  0  0  0  0
%          0  0  0  0  0  0  0  0  0  0  0  0  0
%          0  0  0  0  0 -1  0  0  0  0  0  0  0
%          0  0  0  0  0 -1  0  0  0  0  0  0  0
%          0  0  0  0  0  0  0  0  0 -1  0  0  0
%          0  0  0  0  0  0  0  1 -1  0  0  0  0
%          0  0  0  0  0  0  0 -1  1  0  0  0  0
%          0  0  0  0  0  0  0  0  0  1  0  0  0
%          0  0  0  0  0  0  0  0  0  0  0  0  0
%          0  0  0  0  0  0  0  0  0  0  0  0  0
%          0  0  0  0  0  0  0  0  0  0  0  0  0];
% 
% f = getFeatures;
% player = 1;
% c = countpattern(board,player,f);
% jj = find(c);
% for j = 1:numel(jj)
%     k = find(f.group == jj(j),1);
%     fprintf('%d matches for this feature (modulo symmetries):\n',c(jj(j)))
%     disp(f.pattern{k})
% end