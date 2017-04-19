function [ss as rs isRandom] = rollOut(w,f,epsilon)

n = 13;
b = zeros(n,n);

is_over = 0;
p = 1;

% for recording the (s,a,r)-tuples:
as = NaN(n*n,2);
ss = NaN(n*n,n*n + 1);
rs = zeros(n*n,1);

% for visualisation:
isRandom = zeros(n*n,1);

m = 0;
while ~is_over
    if ~any(b(:) == 0)
        % draw
        break
    end
    
    % move:
    s = [b(:)' p];
    [a isRand] = qAgent(s,w,f,epsilon);
    m = m + 1;
    ss(m,:) = s;
    as(m,:) = a;
    isRandom(m,:) = isRand;
        
    % check win:
    b(a(1),a(2)) = p;
    isWin = checkWin(b,a);
    
    if isWin
        is_over = 1;
        rs(m) = 1; % r
    end
        
    % next player's turn
    p = -p;
end
ss = ss(1:m,:);
as = as(1:m,:);
rs = rs(1:m,:);
isRandom = isRandom(1:m);
