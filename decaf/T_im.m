
load im
img = squeeze(oo(1,:,:,:))-min(oo(:));
imwrite(uint8(img),'in_pubdog.jpg')



for lid = 1:9
load(['ha' num2str(lid)]) 
gg = squeeze(im);
subplot(3,3,lid),imagesc(max(abs(gg),[],3))
end

imwrite(max(abs(gg),[],3),'grad_pubdog.jpg')


imagesc(img+gg)