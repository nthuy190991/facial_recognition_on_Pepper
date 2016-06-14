
fileID = fopen('decimal_wb.txt','r');
sizeA  = [1, 57600];
A = fscanf(fileID,'%u',sizeA);
fclose(fileID);


image = reshape(A, [3,160,120]);
imageR = squeeze(image(1,:,:));
imageG = squeeze(image(2,:,:));
imageB = squeeze(image(3,:,:));

I = uint8(zeros([size(imageR) 3]));
I(:,:,1) = imageR;
I(:,:,2) = imageG;
I(:,:,3) = imageB;

imwrite(I,'Combined2.png');

imwrite(image, 'image.jpg')
figure
imshow(image(:,:,1))

figure
imagesc(image(:,:,2))