function  [ seq ] = action_seq( T, snames, name )
%% Generate data for one action sequence
%   Detailed explanation goes here

% First row of transition matrix T is starting state
curr_state = 1;
seq_actns = [];
seq_states = {};

ncopy = 0;
ndelete = 0;
nmodify = 0;
nemail = 0;
nterminate = 0;
pathlen = 0;

actstr = '';
i = 1;
while curr_state ~= 6

    curr_stateX = mnrnd(1,T(curr_state,:),1);
    curr_state = find(curr_stateX==1);
    
    switch curr_state
        case {2}            % copy
            ncopy = ncopy+1;
            pathlen = pathlen+1;
        case {3}            % delete
            ndelete = ndelete+1;
        case {4}            % modify
            nmodify = nmodify+1;
        case {5}            % email
            nemail = nemail+1;
        case {6}            % terminate
            nterminate = nterminate+1;
    end
    
    curr_act_str = snames{curr_state};
    actstr = strcat(actstr,curr_act_str);
    
    seq_actns = [seq_actns; curr_state];
    
    %state = struct();
    %state.ncopy = ncopy;
    %state.pathlen = pathlen;
    %state.ndelete = ndelete;
    %state.nmodify = nmodify;
    %state.nemail = nemail;
    %state.nterminate = nterminate;
    
    state_str = strcat( ...
        num2str(ncopy), '-',num2str(pathlen), ...
        '-',num2str(ndelete), '-',num2str(nmodify), ...
        '-',num2str(nemail), '-',num2str(nterminate));
    seq_states{i} = state_str;  % state;
    
    i = i+1;
    
end;

seq = struct();
seq.name = name;
seq.actions = seq_actns;
seq.actstr = actstr;
seq.states = seq_states;
