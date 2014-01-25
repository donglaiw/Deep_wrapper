
%load st_bd

switch did
case 0
    fns= dir([DD '/train/*.jpg']);
case 2
    fns= dir([DD '/test/*.jpg']);
end

for i=1:numel(fns)
	%imwrite(im_hatt{i},[fns(i).name(1:end-4) '_tt.jpg'])
    %imwrite(im_hat{i},[fns(i).name(1:end-4) '_t.jpg'])	
    if fid==-1
        % decaf
        switch did
        case 0
            imwrite(im_hat{i},[fns(i).name(1:end-4) '_r' num2str(mid) '.jpg'])	
        case 2
            imwrite(im_hatt{i},[fns(i).name(1:end-4) '_r' num2str(mid) '.jpg'])
        end
    else
        switch did
        case 0
            imwrite(im_hat{i},[fns(i).name(1:end-4) '_r' num2str(mid) '_f' num2str(fid) '.jpg'])	
        case 2
            imwrite(im_hatt{i},[fns(i).name(1:end-4) '_r' num2str(mid) '_f' num2str(fid) '.jpg'])
        end
    end
	%gt = mean(cat(3,data.gts{i}{:}),3);
	%imwrite(uint8(255*gt),[fns(i).name(1:end-4) '_g.jpg'])	
	%imwrite(uint8(255*Es{i}),[fns(i).name(1:end-4) '_st.jpg'])
	%imwrite(edge(rgb2gray(test.Is{i})),[fns(i).name(1:end-4) '_sob.jpg'])
	%imwrite(edge(rgb2gray(test.Is{i}),'canny'),[fns(i).name(1:end-4) '_can.jpg'])				
end


