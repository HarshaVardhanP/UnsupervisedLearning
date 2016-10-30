function visualizeImgs(data)
    W = data;
    min_w = min(min(W));
    max_w = max(max(W));
    W = W-min_w;
    W = W./(-min_w+max_w);
    F = zeros(29,29,1,100);
    for i = 1:size(W,2)
        F(1:28,1:28,:,i) = vec2mat(W(:,i),28);
        F(29,:,1,i) = ones(1,29);
        F(:,29,1,i) = ones(29,1);
    end
    figure,
    montage(F)
end