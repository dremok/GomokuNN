function phi_ = phi(s,a,player,features)

phi_ = zeros(numel(features),1);

if s(a(1),a(2)) == 0
    if player == -1,
        s = -s;
    end
    s(a(1),a(2)) = 1; 
    
    for n = 1:numel(features)
        f = features{n};
        
        [d1 d2] = size(f);
        js = 1:(13 - (d1 - 1));
        ks = 1:(13 - (d2 - 1));
        for j = js
            for k = ks
                % get the sub-part of board:
                s_ = s(j+(0:d1-1),k+(0:d2-1));
                
                % pattern match:
                ii = f ~= 2;
                if all(s_(ii) == f(ii)),
                    phi_(n) = phi_(n) + 1;
                end
            end
        end
    end
end

%% Old version of phi which looked only around a
% function phi_ = phi(s,a,player,features)
% 
% phi_ = NaN(numel(features),1);
% 
% if s(a(1),a(2)) == 0
%     if player == -1,
%         s = -s;
%     end
%     s(a(1),a(2)) = 1; 
%     
%     for n = 1:numel(features)
%         f = features{n};
%         
%         [d1 d2] = size(f);
%         js = a(1) - (d1-1):a(1);
%         ks = a(2) - (d2-1):a(2);
%         js = js(js > 0 & js + (d1-1) <= 13);
%         ks = ks(ks > 0 & ks + (d2-1) <= 13);
%         % offsets:
%         js = js - a(1);
%         ks = ks - a(2);
%         % (j,k) <-> place feature with its top left corner on (a(1)+j,a(2)+k),
%         % so it will cover the square (a(1)+j+(0:(d1-1)),a(2)+k+(0:(d2-1))
%         
%         match = 0;
%         for j = js
%             for k = ks
%                 % if the move completes the pattern:
%                 if f(-j + 1,-k + 1) == 1
%                     % get the sub-part of board:
%                     s_ = s(a(1)+j+(0:d1-1),a(2)+k+(0:d2-1));
%                 
%                     % pattern match:
%                     ii = f ~= 2;
%                     if all(s_(ii) == f(ii)),
%                         match = 1;
%                         break
%                     end
%                 end
%                 if match == 1, break; end
%             end
%         end
%         phi_(n) = match;
%     end
% end
% 
% 
