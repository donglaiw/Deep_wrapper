function out=U_im(im,sz)
if numel(sz)==1
    sz = [sz sz];
end
if nargout==0
    imagesc(reshape(im,sz))
else
    out = reshape(im,sz);
end
