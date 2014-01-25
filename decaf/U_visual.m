


fns = dir('test/*.jpg');
for ff= 11:numel(fns)
nn = fns(ff).name;
nn = nn(1:end-4);


load([nn '.mat'])
for i =1:3
	im(1,:,:,i) = flipud(squeeze(im(1,:,:,i)));
end
b = uint8(128+squeeze(im));
imwrite(b,[nn '_im.jpg'])

szs = [55,27,13,13,13];
num = [96,256,384,384,256];
num_c = 12;
num_r = num/num_c;

pre_im = imresize(b,[55,55],'nearest');
for id=1:5
	load(['result/' nn 'conv' num2str(id) '.mat'])
	im=zeros([55*[num_r(id) num_c]],'uint8');
	%im2=zeros([55*[num_r(id) num_c] 3],'uint8');
	for rid=1:num_r(id)
		for cid=1:num_c
			mask = flipud(imresize(squeeze(mat(1,:,:,cid+(rid-1)*num_c)),[55 55],'nearest'));
			mask = mask-min(mask(:));
			%mask = uint8(255*mask/max(mask(:)));
			%mask = uint8(1.6*mask);
			mask = uint8(mask);
			%tmp_im = pre_im;
			%tmp_im(:,:,1) = mask;
			%im2((rid-1)*55+(1:55),(cid-1)*55+(1:55),:) = tmp_im;
			%im((rid-1)*55+(1:55),(cid-1)*55+(1:55)) = flipud(imresize(uint8(squeeze(mat(1,:,:,cid+(rid-1)*num_c))),[55 55],'nearest'));
			im((rid-1)*55+(1:55),(cid-1)*55+(1:55)) = mask;
		end
	end
	%imwrite(im,['conv' num2str(id) '.jpg'])
	imwrite(im,[nn '_2conv' num2str(id) '.jpg'])
	%imwrite(im,[nn '_conv' num2str(id) '.jpg'])
end

end

