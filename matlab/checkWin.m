function isWin = checkWin(board,move)

n = size(board,1);

isWin = checkWinOne(board(move(1),:));
if ~isWin, isWin = checkWinOne(board(:,move(2))); end
if ~isWin, isWin = checkWinOne(diag(board,move(2)-move(1))); end
if ~isWin, isWin = checkWinOne(diag(fliplr(board),(n-move(2)+1)-move(1))); end

function isWin = checkWinOne(x)
y = conv(x,ones(5,1),'same');
isWin = max(abs(y)) == 5;

%% Test it:
% board = [0  0  0  0  0  0  0  0  0  0  0  0  0
%          0  0  0  0  0  0  0  0  0  0  0  0  0
%          0  0  0  0  0  0  0  0  0  0  0  0  0
%          0  0  0  0  0  0  0  0  0  0  0  0  0
%          0  0  0  0  0  0  0  0  0  0  0  0  0
%          0  0  0  0  0  0  0  0  0  0  0  0  0
%          0  0  0  0  0  0  0  0  0  0  0  0  0
%          0  0  0  1  1  1  1  1  0  0  0  0  0
%          0  0  0  0  0  0  0  0  0  0  0  0  0
%          0  0  0  0  0  0  0  0  0  0  0  0  0
%          0  0  0  0  0  0  0  0  0  0  0  0  0
%          0  0  0  0  0  0  0  0  0  0  0  0  0
%          0  0  0  0  0  0  0  0  0  0  0  0  0];     
% checkWin(board,[8 5])
% 
% board = [0  0  0  0  0  0  0  0  0  0  0  0  0
%          0  0  0  0  0  0  0  0  0  0  0  0  0
%          0  0  0  0  0  0  0  0  0  0  0  0  0
%          0  0  0  0  0  0  0  0  0  0  0  0  0
%          0  0  0  0  0  0  0  0  0  0  0  0  0
%          0  0  0 -1  0  0  0  0  0  0  0  0  0
%          0  0  0  0 -1  0  0  0  0  0  0  0  0
%          0  0  0  1  1 -1  1  1  0  0  0  0  0
%          0  0  0  0  0  0 -1  0  0  0  0  0  0
%          0  0  0  0  0  0  0 -1  0  0  0  0  0
%          0  0  0  0  0  0  0  0  0  0  0  0  0
%          0  0  0  0  0  0  0  0  0  0  0  0  0
%          0  0  0  0  0  0  0  0  0  0  0  0  0];     
% checkWin(board,[7 5])
% 
% board = [0  0  0  0  0  0  0  0  0  0  0  0  0
%          0  0  0  0  0  0  0  0  0  0  0  0  0
%          0  0  0  0  0  0  0  0  0  0  0  0  0
%          0  0  0  0  0  0  0  0  0  0  0  1  0
%          0  0  0  0  0  0  0  0  0  0  1  0  0
%          0  0  0 -1  0  0  0  0  0  1  0  0  0
%          0  0  0  0 -1  0  0  0  1  0  0  0  0
%          0  0  0  1  1  0  1  1  0  0  0  0  0
%          0  0  0  0  0  0 -1  0  0  0  0  0  0
%          0  0  0  0  0  0  0 -1  0  0  0  0  0
%          0  0  0  0  0  0  0  0  0  0  0  0  0
%          0  0  0  0  0  0  0  0  0  0  0  0  0
%          0  0  0  0  0  0  0  0  0  0  0  0  0];     
% checkWin(board,[4 12])
% 
% board = [0  0  0  0  0  0  0  0  0  0  0  0  0
%          0  0  0  0  0  0  0  0  0  0  0  0  0
%          0  0  0  0  0  0  0  0  0  0  0  0  0
%          0  0  0  0  0  0  0  0  0  0  0  1  0
%          0  0  0  0 -1  0  0  0  0  0  1  0  0
%          0  0  0 -1  0  0  0  0  0  1  0  0  0
%          0  0 -1  0 -1  0  0  0  0  0  0  0  0
%          0 -1  0  1  1  0  1  1  0  0  0  0  0
%         -1  0  0  0  0  0 -1  0  0  0  0  0  0
%          0  0  0  0  0  0  0 -1  0  0  0  0  0
%          0  0  0  0  0  0  0  0  0  0  0  0  0
%          0  0  0  0  0  0  0  0  0  0  0  0  0
%          0  0  0  0  0  0  0  0  0  0  0  0  0];     
% checkWin(board,[9 1])