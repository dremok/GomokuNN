function c = countpattern(b, p, f)

% b: board
% p: player 
% f: features

c = NaN(size(f,1),1);
for j = 1:size(f,1)
    c(j) = countpat(b*p,f{j});
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
%     fprintf('%d matches for this feature:\n',c(jj(j)))
%     disp(f{jj(j)})
% end