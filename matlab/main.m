
n = 13;
board   = zeros(n,n);
[f w f_group]   = getFeatures;      % features for computing phi(s,a)
epsilon = 0.1;              % for epsilon-greedy strategy

% ADAM pars (https://arxiv.org/pdf/1412.6980.pdf):
adam.a = 0.001;
adam.b1 = 0.9;
adam.b2 = 0.999;
adam.eps = 1e-8;
adam.n   = 0;
adam.m   = zeros(size(w));
adam.v   = zeros(size(w));

%%

wAll = [];
LAll = [];

% initialise experience memory: 
N = 10000;
clear D
D.s       = NaN(N,n*n + 1);
D.phi     = NaN(N,1+100*(numel(w)+2));
D.a       = NaN(N,2);
D.r       = NaN(N,1);
D.sNext   = NaN(N,n*n + 1);
D.phiNext = NaN(N,1+100*(numel(w)+2));
D.n       = 0;
D.m       = 0;
%%

for kk = 1:100000
    %% rollout
    fprintf('Rollout %d ..'); tic; [ss as rs isRandom phiAll] = rollOut(w,f,epsilon); toc
    
    %% add experience to D
    
    ii = D.n + (1:size(ss,1));
    D.s(ii,:)     = [ss(1:2:end-2,:); ss(2:2:end-2,:);  ss(end-1,:);    ss(end,:)];
    D.phi(ii,:)   = [phiAll(1:2:end-2,:); phiAll(2:2:end-2,:);  phiAll(end-1,:);    phiAll(end,:)];
    D.a(ii,:)     = [as(1:2:end-2,:); as(2:2:end-2,:);  as(end-1,:);    as(end,:)];
    D.r(ii,:)     = [rs(1:2:end-2,:); rs(2:2:end-2,:); -rs(end,:);      rs(end,:)];
    D.sNext(ii,:) = [ss(3:2:end,:);   ss(4:2:end,:);    NaN(1,n*n + 1); NaN(1,n*n + 1)];
    D.phiNext(ii,:)   = [phiAll(3:2:end,:); phiAll(4:2:end,:);  NaN(1,size(phiAll,2));    NaN(1,size(phiAll,2))];
    
    if D.n + numel(ii) < N
        D.n = D.n + numel(ii);
    else
        % memory almost full:
        D.n = 0;
    end
    D.m = max(D.m,D.n);
    
    %% stochastic gradient descent
    
    mini_batch_size = 50;
    fprintf('Minibatch '); tic
    for mb = 1:5
        fprintf('%d',mb)
        ii = randi(D.m,mini_batch_size,1);
        
        dLdw  = zeros(size(w));
        L = 0;
        for i = 1:numel(ii)
            j = ii(i);
            s = D.s(j,:);
            a = D.a(j,:);
            r = D.r(j,:);
            sNext = D.sNext(j,:);
            
            if isnan(sNext(1))
                % we're looking at a final state
                y = r;
            else
                nActions = D.phiNext(j,1);
                tmp = reshape(D.phiNext(j,1+(1:nActions*(2+numel(w)))),2+numel(w),nActions);
                phi_ = tmp(3:end,:);
                Q = w'*phi_;
                k = find(Q == max(Q));
                if numel(k) > 1, k = k(randi(numel(k),1,1)); end
                y = Q(k);
            end
            phi_sa = phi(s,a,f);
            dLdw = dLdw + (y - w'*phi_sa)*phi_sa;
            L    = L + (y - w'*phi_sa)^2;
        end
        L = -L / mini_batch_size;
        dLdw = dLdw / mini_batch_size;
        adam.n = adam.n + 1;
        
        % ADAM:
        adam.m = adam.b1*adam.m + (1-adam.b1)*dLdw;
        adam.v = adam.b2*adam.v + (1 - adam.b2)*dLdw.^2;
        mHat = adam.m / (1 - adam.b1^adam.n);
        vHat = adam.v / (1 - adam.b2^adam.n);
%         w = w - adam.a * mHat ./ (sqrt(vHat) + adam.eps);
        w = w + 0.001 * mHat ./ (sqrt(vHat) + adam.eps);
        
        for mm = 1:max(f_group)
           jj = f_group == mm;
           w(jj) = mean(w(jj));
        end
        
%         w = w + 0.2*dLdw;
        wAll(end+1,:) = w;
        LAll(end + 1) = L;
    end
    toc
    
    if mod(kk,5000) == 0
        tnow = now;
        save dump D w tnow
    end
end

%% visualize:

token = {'o','*'};
p = 1;
figure; hold on; axis([1 13 1 13]); grid on; 
for j = 1:size(as,1)
   plot(as(j,1),as(j,2),token{p+1});
   
   if isRandom(j), plot(as(j,1),as(j,2),['g' token{p+1}]); end
   
   drawnow;
   pause;
   
   p = mod(p + 1,2);
end
