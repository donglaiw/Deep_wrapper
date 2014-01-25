DD= '/home/Stephen/Desktop/Data/Seg/BSR/BSDS500/data/images/';
	psz= 227;
	psz1= 55;
	%im_pst = zeros(psz1^2*3,num,'uint8');	
	%im_pstt = zeros(psz1^2*3,num,'uint8');	

if ~exist('train','var')
    train =load('/home/Stephen/Desktop/Deep_Low/occ/dn_ucb0');
end
if ~exist('valid','var')
    test =load('/home/Stephen/Desktop/Deep_Low/occ/dn_ucb1'); 
end
if ~exist('test','var')
    test =load('/home/Stephen/Desktop/Deep_Low/occ/dn_ucb2'); 
end

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
	gt_ps = zeros(psz1^2,num,'single');
    cc = zeros(1,4);
    %im_ps = im_pst;
	for i=1:num
            sz = size(data.Is{i});
            cc([1 3]) = 1;
            cc([2 4]) = sz(1:2);
			gt = zeros(psz1);
            for j=1:numel(data.gts{i})
                gt = gt + double(U_seg2bd(double(data.segs{i}{j}),cc,[psz1 psz1]));
            end
			gt_ps(:,i) = reshape(gt/numel(data.gts{i}),[],1);
			%im_ps(:,i) = reshape(imresize(data.Is{i}(1:psz,1:psz,:),[psz1 psz1]),[],1);
        end
    switch did
    case 0
        gt_pst = gt_ps;
        save conv_t3 gt_pst
    case 1
        gt_psv = gt_ps;
        save conv_v3 gt_psv
    case 2
        gt_pstt = gt_ps;
        save conv_tt3 gt_pstt
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
decaf_t=load('decaf_3_0','conv1');
sz = size(decaf_t.conv1);
conv12 = reshape(decaf_t.conv1,[sz(1)*sz(2),sz(3)*sz(4),sz(5)]);
mat_x = reshape(permute(conv12,[2 1 3]),[],sz(end));
mat_y = reshape(gt_pst,[],1);
w= mat_x\(mat_y);
yhat = mat_x*w;
yhat2 = reshape(yhat,[psz1^2,sz(1)]);
% 548.4589 
mean(sum(reshape(sum(abs(yhat-mat_y),2),psz1^2,[]),1))

save train_w3 w

im_hat = cell(1,sz(1));
for i=1:sz(1)
	im_hat{i} = imresize(U_im(yhat2(:,i),psz1),size(train.gts{i}{1}));
end
save bd_decaf_3_0 im_hat

decaf_tt=load('decaf_3_2','conv1');
sz = size(decaf_tt.conv1);
conv12_t = reshape(decaf_tt.conv1,[sz(1)*sz(2),sz(3)*sz(4),sz(5)]);
mat_xx = reshape(permute(conv12_t,[2 1 3]),[],sz(end));
mat_yy = reshape(gt_pstt,[],1);
yhatt = mat_xx*w;
yhatt2 = reshape(yhatt,[psz1^2,sz(1)]);
mean(sum(reshape(sum((yhatt-mat_yy).^2,2),psz1^2,[]),1))
% 394.1493
im_hatt = cell(1,sz(1));
for i=1:sz(1)
	im_hatt{i} = imresize(U_im(yhatt2(:,i),psz1),size(test.gts{i}{1}));
end
save bd_decaf_3_2 im_hatt

