% =====================================
% \author  Yitian Shao
% \created 11/19/2015 
% 
% Used to display extracted depth image
% =====================================
IMAGE_WIDTH = 960;
IMAGE_HEIGHT =540;

% fileID = fopen('originalMap.txt');
fileID = fopen('modifedMap.txt');
% fileID = fopen('diffX.txt');
% fileID = fopen('diffY.txt');

dataStream = textscan(fileID,repmat('%f',1,IMAGE_WIDTH));
fclose(fileID);

depthMatrix = zeros(IMAGE_HEIGHT,IMAGE_WIDTH);

for i = 1:IMAGE_HEIGHT
    for j = 1:IMAGE_WIDTH
        depthMatrix(i,j) = dataStream{j}(i);
    end
end

%%
figure('Position',[400,200,IMAGE_WIDTH,IMAGE_HEIGHT-26])
surf(depthMatrix,'EdgeColor','none');
view(2);
xlim([0 IMAGE_WIDTH]);
ylim([0 IMAGE_HEIGHT]);
axis equal
colorbar