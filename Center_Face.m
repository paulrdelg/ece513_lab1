% Michael Greer
% Code to center all of the faces in the frame

% WARNING: This code overwrites the original images, be sure to make copies

clc
clear

% Change this to the relative path for the images folder
imagedir = "Faces/yalefaces";

imagefiles = dir(fullfile(imagedir,"*.gif"));

h = 243;
w = 320;

for f = 1:length(imagefiles)
   
    file = fullfile(imagedir, imagefiles(f).name);
    im = imread(file);
    
    figure;
    imagesc(im);
    [x,y] = ginput(1);
    
    x_shift = cast(-1*(x-(w/2)), 'int32');
    y_shift = cast(-1*(y-(h/2)), 'int32');
    
    im = circshift(im, x_shift,2);
    im = circshift(im, y_shift,1);
    
    imwrite(im, file);
    
    close
    
end