DD= '/home/Stephen/Desktop/Data/Seg/BSR/BSDS500/data/images/';

	psz= 227;
	psz1= 55;
	gt_pst = zeros(4,psz1^2,num,'single');
	gt_pstt = zeros(4,psz1^2,num,'single');
	im_ps = zeros(4,psz1^2*3,num,'uint8');
	
	dotrain = 0;
	if dotrain
		%train =load('/home/Stephen/Desktop/Deep_Low/occ/dn_ucb1');
		data = train;
	else
		%test =load('/home/Stephen/Desktop/Deep_Low/occ/dn_ucb2'); 
		data = test;
	end
 	
	for i=1:num
			gt = mean(cat(3,data.gts{i}{:}),3);
			gt_ps(1,:,i) = reshape(imresize(gt(1:psz,1:psz),[psz1 psz1]),[],1);
			gt_ps(2,:,i) = reshape(imresize(gt(1:psz,(end-psz+1):end),[psz1 psz1]),[],1);
			gt_ps(3,:,i) = reshape(imresize(gt((end-psz+1):end,1:psz),[psz1 psz1]),[],1);
			gt_ps(4,:,i) = reshape(imresize(gt((end-psz+1):end,(end-psz+1):end),[psz1 psz1]),[],1);			
			
			gt_ps(1,:,i) = gt_ps(1,:,i)*max(max(gt(1:psz,1:psz)))/max(gt_ps(1,:,i));
			gt_ps(2,:,i) = gt_ps(2,:,i)*max(max(gt(1:psz,(end-psz+1):end)))/max(gt_ps(2,:,i));
			gt_ps(3,:,i) = gt_ps(3,:,i)*max(max(gt((end-psz+1):end,1:psz)))/max(gt_ps(3,:,i));
			gt_ps(4,:,i) = gt_ps(4,:,i)*max(max(gt((end-psz+1):end,(end-psz+1):end)))/max(gt_ps(4,:,i));

			%{
			im_ps(1,:,i) = reshape(imresize(data.Is{i}(1:psz,1:psz,:),[psz1 psz1]),[],1);
			im_ps(2,:,i) = reshape(imresize(data.Is{i}(1:psz,(end-psz+1):end,:),[psz1 psz1]),[],1);
			im_ps(3,:,i) = reshape(imresize(data.Is{i}((end-psz+1):end,1:psz,:),[psz1 psz1]),[],1);
			im_ps(4,:,i) = reshape(imresize(data.Is{i}((end-psz+1):end,(end-psz+1):end,:),[psz1 psz1]),[],1);
			%}		
	end
if dotrain
	gt_pst = gt_ps;
else
	gt_pstt = gt_ps;
end


save conv_pt gt_pstt
save conv_p gt_pst im_ps



decaf_t=load('conv1_1');

%{
imagesc(uint8(reshape(im_ps(2,:,1),[psz1 psz1 3]))) 
imagesc(reshape(gt_ps(2,:,1),[psz1 psz1 ])) 
%}

sz = size(decaf_t.mats);
mats2 = reshape(decaf_t.mats,[sz(1)*sz(2),sz(3)*sz(4),sz(5)]);

%{
% demo: one case
mat_x = squeeze(mats2(1,:,:));
mat_y = gt_pst(1,:,1);
w= mat_x\(mat_y');
yhat = mat_x*w;
imagesc(reshape(yhat,[psz1 psz1]))
imagesc(reshape(mat_y,[psz1 psz1]))

imwrite(reshape(im_ps(1,:,1),[psz1 psz1 3]),'reg0_im.jpg')
imwrite(reshape(mat_y,[psz1 psz1]),'reg0_bd.jpg')
[wmax,wid] = sort(w,'descend');

for i =1:8
	ind = (i-1)*12+(1:12);
	imwrite(uint8(reshape(mat_x(:,wid(ind)), [psz1 psz1*12])),['reg0_f' num2str(i) '.jpg']);
	%dlmwrite('reg0_score.txt',wmax(ind)','-append')
	end

for i =1:8
	ind = (i-1)*12;
	im = zeros(psz1,psz1*12);
	for id=1:12
		im(:,(id-1)*psz1+(1:psz1)) = reshape(mat_x(:,wid(ind+id)), [psz1 psz1]);
		im(:,(id-1)*psz1+(1:psz1)) = im(:,(id-1)*psz1+(1:psz1)) /max(max(im(:,(id-1)*psz1+(1:psz1))));
	end
	imwrite(uint8(255*im),['reg0_ff' num2str(i) '.jpg']);
	%dlmwrite('reg0_score.txt',wmax(ind)','-append')
	end
imwrite(uint8(255*reshape(yhat,[psz1 psz1])),'reg0_yhat.jpg')
%}

mat_x = reshape(permute(mats2,[2 1 3]),[],sz(end));
mat_y = reshape(permute(gt_pst,[2 3 1]),[],1);
w= mat_x\(mat_y);
yhat = mat_x*w;
yhat2 = reshape(yhat,[psz1^2,sz(1),4]);
% 212.9404
mean(sum(reshape(sum(abs(yhat-mat_y),2),psz1^2,[]),1))

save train_w w


im_hat = cell(1,sz(1));
for i=1:sz(1)
	sszz =size(train.Is{i});
	im = zeros(sszz(1:2));
	cc = zeros(sszz(1:2));
	tmp = imresize(reshape(yhat2(:,i,1),[psz1 psz1]),[psz psz]);
	im(1:psz,1:psz) = im(1:psz,1:psz) + max(yhat2(:,i,1))*tmp/max(tmp(:));

	tmp = imresize(reshape(yhat2(:,i,2),[psz1 psz1]),[psz psz]);
	im(1:psz,(end-psz+1):end) = im(1:psz,(end-psz+1):end) + max(yhat2(:,i,2))*tmp/max(tmp(:));
	
	tmp = imresize(reshape(yhat2(:,i,3),[psz1 psz1]),[psz psz]);
	im((end-psz+1):end,1:psz) = im((end-psz+1):end,1:psz) + max(yhat2(:,i,3))*tmp/max(tmp(:));
	
	tmp = imresize(reshape(yhat2(:,i,4),[psz1 psz1]),[psz psz]);
	im((end-psz+1):end,(end-psz+1):end) = im((end-psz+1):end,(end-psz+1):end) + max(yhat2(:,i,4))*tmp/max(tmp(:));

	cc(1:psz,1:psz) = cc(1:psz,1:psz) + 1;
	cc(1:psz,(end-psz+1):end) = cc(1:psz,(end-psz+1):end) + 1;
	cc((end-psz+1):end,1:psz) = cc((end-psz+1):end,1:psz) + 1;
	cc((end-psz+1):end,(end-psz+1):end) = cc((end-psz+1):end,(end-psz+1):end) + 1;
	im_hat{i} = im./cc;
end


decaf_tt=load('conv1_2');
sz = size(decaf_tt.mats);
mats2_t = reshape(decaf_tt.mats,[sz(1)*sz(2),sz(3)*sz(4),sz(5)]);
mat_xx = reshape(permute(mats2_t,[2 1 3]),[],sz(end));
mat_yy = reshape(permute(gt_pst,[2 3 1]),[],1);
yhatt = mat_xx*w;
yhatt2 = reshape(yhatt,[psz1^2,sz(1),4]);
mean(sum(reshape(sum((yhatt-mat_yy).^2,2),psz1^2,[]),1))

% 276.3612
im_hatt = cell(1,sz(1));
for i=1:sz(1)
	sszz =size(test.Is{i});
	im = zeros(sszz(1:2));
	cc = zeros(sszz(1:2));
	im(1:psz,1:psz) = im(1:psz,1:psz) + imresize(reshape(yhatt2(:,i,1),[psz1 psz1]),[psz psz]);
	im(1:psz,(end-psz+1):end) = im(1:psz,(end-psz+1):end) + imresize(reshape(yhatt2(:,i,2),[psz1 psz1]),[psz psz]);
	im((end-psz+1):end,1:psz) = im((end-psz+1):end,1:psz) + imresize(reshape(yhatt2(:,i,3),[psz1 psz1]),[psz psz]);
	im((end-psz+1):end,(end-psz+1):end) = im((end-psz+1):end,(end-psz+1):end) + imresize(reshape(yhatt2(:,i,4),[psz1 psz1]),[psz psz]);
	cc(1:psz,1:psz) = cc(1:psz,1:psz) + 1;
	cc(1:psz,(end-psz+1):end) = cc(1:psz,(end-psz+1):end) + 1;
	cc((end-psz+1):end,1:psz) = cc((end-psz+1):end,1:psz) + 1;
	cc((end-psz+1):end,(end-psz+1):end) = cc((end-psz+1):end,(end-psz+1):end) + 1;
	im_hatt{i} = im./cc;
end


load st_bd
fns= dir([DD '/test/*.jpg']);
fns= dir([DD '/train/*.jpg']);
for i=1:numel(fns)
	%imwrite(im_hatt{i},[fns(i).name(1:end-4) '_tt.jpg'])
	%imwrite(uint8(255*im_hat{i}),[fns(i).name(1:end-4) '_t.jpg'])	
	%gt = mean(cat(3,data.gts{i}{:}),3);
	%imwrite(uint8(255*gt),[fns(i).name(1:end-4) '_g.jpg'])	
	%imwrite(uint8(255*Es{i}),[fns(i).name(1:end-4) '_st.jpg'])
	imwrite(edge(rgb2gray(test.Is{i})),[fns(i).name(1:end-4) '_sob.jpg'])
	imwrite(edge(rgb2gray(test.Is{i}),'canny'),[fns(i).name(1:end-4) '_can.jpg'])				
end


