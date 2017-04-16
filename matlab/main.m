
board   = zeros(13,13);
[f w]   = getFeatures;      % features for computing phi(s,a)
epsilon = 0.1;              % for epsilon-greedy strategy

%% testing phi:

board = [0  0  1  1  0  0  0  0  0  0  0  0  0
         0  1  0  0  0  0  0  0  0  0  0  0  0
         0  0  0  0  0  0  0  0  0  0  0  0  0
         0  0  0  0  0  0  0  0  0  0  0  0  0
         0  0  0  0  0 -1  0  0  0  0  0  0  0
         0  0  0  0  0 -1  0  0  0  0  0  0  0
         0  0  0  0  0  0  0  0  0 -1  0  0  0
         0  0  0  0  0  0  0  1 -1  0  0  0  0
         0  0  0  0  0  0  0 -1  1  0  0  0  0
         0  0  0  0  0  0  0  0  0  1  0  0  0
         0  0  0  0  0  0  0  0  0  0  0  0  0
         0  0  0  0  0  0  0  0  0  0  0  0  0
         0  0  0  0  0  0  0  0  0  0  0  0  0];

player = 11;
a = [2 3];

clc
fprintf('board after move (%d,%d) for player %d:\n',a(1),a(2),player);
board_ = board;
board_(a(1),a(2)) = player;
disp(board_)  

phi_ = phi(board,[2 3],player,f);
jj = find(phi_);
for j = 1:numel(jj)
    fprintf('%d matches for this feature:\n',phi_(jj(j)))
    disp(f{jj(j)})
end

%% rollout:

board = zeros(n,n);

is_over = 0;
player  = 1;
moves = NaN(n*n,2);
nMoves = 0;
while 1
    if ~any(board(:) == 0)
        winner  = 0; % draw
        break
    end
    
    % move:
    move   = qAgent(board,player,w,f,epsilon);
    nMoves = nMoves + 1;
    moves(nMoves,:) = move;
    board(move(1),move(2)) = player;
    
    % check win:
    isWin = checkWin(board,move);
    if isWin
        winner = player;
        break
    end
    
    % next player's turn
    player = -player;
end
moves = moves(1:nMoves,:);

%% visualize:

token = {'o','x'};
p = 1;
figure; hold on; axis([1 13 1 13]); grid on; 
for j = 1:size(moves,1)
   plot(moves(j,1),moves(j,2),token{p+1});
   
   drawnow;
   pause;
   
   p = mod(p + 1,2);
end
