
n = 13;
board   = zeros(n,n);
[f w]   = getFeatures;      % features for computing phi(s,a)
epsilon = 0.1;              % for epsilon-greedy strategy

%%

wAll = w';

% initialise experience memory: 
N = 10000;
clear D
D.board     = NaN(N,n*n);
D.player    = NaN(N,1);
D.move      = NaN(N,2);
D.reward    = NaN(N,1);
D.boardNext = NaN(N,n*n);
D.moveNext  = NaN(N,2);
D.n         = 0;

%%

for kk = 1:100
    %% rollout
    fprintf('Rollout %d ..'); tic
    [states moves rewards] = rollOut(w,f,epsilon);
    toc
    
    %% add experience to D
    
    s     = [states(1:2:end-2,:);  states(2:2:end-2,:);  states(end-1,:); states(end,:)];
    a     = [moves(1:2:end-2,:);   moves(2:2:end-2,:);   moves(end-1,:);   moves(end,:)];
    r     = [rewards(1:2:end-2,:); rewards(2:2:end-2,:);             -1; rewards(end,:)];
    sNext = [states(3:2:end,:);    states(4:2:end,:);    NaN(1,n*n + 1); NaN(1,n*n + 1)];
    aNext = [moves(3:2:end,:);     moves(4:2:end,:);     NaN(1,2);             NaN(1,2)];
    
    if D.n + size(states,1) < size(D.board,1)
        ii = D.n + (1:size(s,1));
        jj = 1:size(s,1);
        D.n = D.n + numel(jj);
    else
        % memory almost full:
        ii = (D.n + 1):D.N;
        jj = size(s,1) - numel(ii) + 1:size(s,1);
        D.n = 0;
    end
    D.board(ii,:)       = s(jj,1:end-1);
    D.player(ii)        = s(jj,end);
    D.move(ii,:)        = a(jj,:);
    D.reward(ii)        = r(jj);
    D.boardNext(ii,:)   = sNext(jj,1:end-1);
    D.moveNext(ii,:)    = aNext(jj,:);
    
    %% stochastic gradient descent
    
    mini_batch_size = 50;
    fprintf('Minibatch '); tic
    for mb = 1:5
        fprintf('%d',mb)
        ii = randi(D.n,mini_batch_size,1);
        
        dLdw  = zeros(size(w));
        for i = 1:numel(ii)
            j = ii(i);
            if D.reward(j) ~= 0
                Qmax = D.reward(j);
            else
                boardn = D.boardNext(j,:);
                an = D.moveNext(j,:);
                Qmax = w'*phi(reshape(boardn,13,13),an,D.player(j),f);
            end
            board = reshape(D.board(j,:),13,13);
            a = D.move(j,:);
            phi_sa = phi(board,a,D.player(j),f);
            dLdw = dLdw + (Qmax - w'*phi_sa)*phi_sa;
        end
        
        dLdw = dLdw / mini_batch_size;
            
        w = w + 0.1*dLdw;
        wAll(end+1,:) = w;
    end
    toc
end

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
