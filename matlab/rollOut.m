function [states moves rewards] = rollOut(w,f,epsilon)

n = 13;
board = zeros(n,n);

is_over = 0;
player  = 1;

% for recording the (s,a,r)-tuples:
moves   = NaN(n*n,2);
states  = NaN(n*n,n*n + 1);
rewards = zeros(n*n,1);

nMoves = 0;
while ~is_over
    if ~any(board(:) == 0)
        winner  = 0; % draw
        break
    end
    
    % move:
    move   = qAgent(board,player,w,f,epsilon);
    nMoves = nMoves + 1;
    states(nMoves,:) = [board(:)' player]; % s
    moves(nMoves,:)  = move;               % a
    
    % check win:
    board(move(1),move(2)) = player;
    isWin = checkWin(board,move);
    
    if isWin
        winner = player;
        is_over = 1;
        rewards(nMoves) = 1; % r
    end
        
    % next player's turn
    player = -player;
end
states  = states(1:nMoves,:);
moves   = moves(1:nMoves,:);
rewards = rewards(1:nMoves,:);