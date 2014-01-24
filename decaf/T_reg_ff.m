DD= '/home/Stephen/Desktop/Data/Seg/BSR/BSDS500/data/images/';
	psz= 227;
	psz1= 55;

if ~exist('train','var')
    train =load('/home/Stephen/Desktop/Deep_Low/occ/dn_ucb0');
end
if ~exist('valid','var')
    valid =load('/home/Stephen/Desktop/Deep_Low/occ/dn_ucb1'); 
end
if ~exist('test','var')
    test =load('/home/Stephen/Desktop/Deep_Low/occ/dn_ucb2'); 
end

mid = 2;
fid=0;
for did = [0 2]
    switch did
    case 0
		data = train;
    case 1
		data = valid;
    case 2
		data = test;
	end 	
    num = numel(data.gts);
	im_ps = zeros([psz psz num],'single');
    cc = zeros(1,4);
    %im_ps = im_pst;
	for i=1:num
        switch mid
        case 1
        case 2
            sz = size(data.Is{i});
            cc([1 3]) = floor((sz(1:2)-psz)/2)+1;
            cc([2 4]) = cc([1 3])+psz-1;
			im_ps(:,:,i) = data.Is{i}(cc(1):cc(2),cc(3):cc(4));
        case 3
        end

    end
    switch did
    case 0
        im_pst = im_ps;
    case 1
        im_psv = im_ps;
    case 2
        im_pstt = im_ps;
    end
end
%{
for ii=0:2
    load(['decaf_' num2str(ii)]);
    for jj=1:5
        eval(['conv' num2str(jj) '=conv' num2str(jj) '(:,1,:,:,:);'])
    end
    save(['decaf_' num2str(ii)],'conv1','conv2','conv3','conv4','conv5');
end
%}
load(['conv_t' num2str(mid)])
load(['conv_tt' num2str(mid)])
ff= U_f_LM(11);
num_train = numel(train.Is);
conv12 = zeros([num_train psz1^2 size(ff,3)]);
for i = 1:num
    for j = 1:size(ff,3)
        tmp = conv2(double(im_pst(:,:,i)),ff(:,:,j),'valid');
        conv12(i,:,j) = reshape(tmp(1:4:end,1:4:end),[],1);
    end
end
thres = 5;
conv12= abs(conv12);
conv12(conv12<thres) = 0;

sz = size(conv12);
mat_x = [reshape(permute(conv12,[2 1 3]),[],sz(end)) ones(num_train*psz1^2,1)];
mat_y = reshape(gt_pst,[],1);
w= mat_x\(mat_y);
yhat = mat_x*w;
yhat2 = reshape(yhat,[psz1^2,sz(1)]);
% 548.4589 
%{
id=2;
U_im(gt_pst(:,id),psz1)
U_im(conv12(id,:,1),psz1)

%}
mean(sum(reshape(sum(abs(yhat-mat_y),2),psz1^2,[]),1))

save(['train_w' num2str(mid) '_' num2str(fid)],'w')

im_hat = cell(1,num_train);
for i=1:num_train
	im_hat{i} = imresize(U_im(yhat2(:,i),psz1),[psz psz]);
end
%find(arrayfun(@(x) strcmp(fns(x).name(1:4),'2018'),1:100)==1)

num_test = numel(test.Is);
conv12_t = zeros([num_test psz1^2 size(ff,3)]);
for i = 1:num
    for j = 1:size(ff,3)
        tmp = conv2(double(im_pstt(:,:,i)),ff(:,:,j),'valid');
        conv12_t(i,:,j) = reshape(tmp(1:4:end,1:4:end),[],1);
    end
end
conv12_t= abs(conv12_t);
conv12_t(conv12_t<thres) = 0;
mat_xx = [reshape(permute(conv12_t,[2 1 3]),[],sz(end)) ones(num_test*psz1^2,1)];
mat_yy = reshape(gt_pstt,[],1);
yhatt = mat_xx*w;
yhatt2 = reshape(yhatt,[psz1^2,sz(1)]);
mean(sum(reshape(sum((yhatt-mat_yy).^2,2),psz1^2,[]),1))
% 394.1493
im_hatt = cell(1,num_test);
for i=1:num_test
	im_hatt{i} = imresize(U_im(yhatt2(:,i),psz1),[psz psz]);
end
save(['bd_f' num2str(fid)],'im_hat','im_hatt')

