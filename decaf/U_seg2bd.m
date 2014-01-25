function y= U_seg2bd(seg,crop,resize)

if exist('crop','var')
    seg = seg(crop(1):crop(2),crop(3):crop(4));
end
if exist('resize','var')
    seg = imresize(seg,resize,'nearest');
end

[g1,g2]=gradient(double(seg));
y= (abs(g1)+abs(g2))>0;
