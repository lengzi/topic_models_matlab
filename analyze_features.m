%%

% Assume that state2id, id2state, Tau, Theta are loaded

[~, IX] = sort(Tau(:,1),'descend');

for i=1:10
    %fprintf('Index: %d\n',IX(i,1));
    val = id2state.get(IX(i));
    disp(val);
end
