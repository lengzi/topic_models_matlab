function  [ actns ] = action_seq_user( n, P, T, snames, name )
%% Generate data for one action sequence
%   Detailed explanation goes here

actns = {};

for i=1:n

    curr_TX = mnrnd(1,P,1);
    curr_T = find(curr_TX==1);
    
    act_name = strcat(name,num2str(i));
    
    actns{i} = action_seq(T{curr_T}, snames, act_name);

end
