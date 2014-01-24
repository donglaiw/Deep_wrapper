
load /home/Stephen/Desktop/Deep_low/occ/dn_ucb2

DATA_UCB = '/home/Stephen/Desktop/Data/Seg/BSR/BSDS500/data/';
tstImgDir = [DATA_UCB '/images/test/'];
tstGtDir = [DATA_UCB '/groundTruth/test/'];
imgIds=dir([tstImgDir '*.jpg']);
lname = textread('label.txt','%s','delimiter','\n');
lname{sid(100:100:end)}


im_id= find(arrayfun(@(x) strcmp(imgIds(x).name,'100099.jpg'),1:200))
%{
imagesc(Is{im_id})
for i=1:5
	subplot(3,2,i),imagesc(segs{im_id}{i})
end
%}

mask = segs{im_id}{1}==4;
%imagesc(mask)


INPUT_DIM = 227;
margin = floor((size(mask)-INPUT_DIM)/2);
mask = mask(margin(1)+(1:INPUT_DIM),margin(2)+(1:INPUT_DIM)); 

bp_val = zeros([size(mask) 3 1000],'single');
parfor lid = 1:1000
tmp = load(['ha' num2str(lid-1)]) 
bp_val(:,:,:,lid) = squeeze(tmp.im);
end
save bsd_1_bp bp_val



mask_score = mask;
mask_score(mask_score==0) = -1;
bp_mask = squeeze(max(abs(bp_val),[],3));


thres = 0.1;
score = sum(reshape(bsxfun(@times,double(bp_mask>thres),mask_score),[],1000));
[~,sid]=sort(score,'ascend');
for i=1:9
	pid = sid(100*(i+1));
	subplot(3,3,i),imagesc(bp_mask(:,:,pid)>0.2),title([lname{pid} ' ' num2str(score(pid))])
end