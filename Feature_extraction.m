%% extract features from original optical images
imds = imageDatastore ...
    (fullfile('C:\Users\lizeb\Box\research projects\Images for Deep Learning\', ...
    {'new images 4.4.2020'}),'LabelSource','foldernames', 'FileExtensions', {'.tif'});
count = countEachLabel(imds);
count1 = table2array(count(1,2))

TotalfeatureVector = [];
for i = 1:count1
    img = readimage(imds, i);
    R = img(:,:,1)
    entropy_1 = entropy(R) % extract entroy
    [m,n] = size(R)
    G = img(:,:,2)
     for i = 1:m
      for j = 1:n
       if R(i,j) > 190
          R(i,j) = 190;
       if G(i,j) < 206  %manully tuned for lightspots
          G(i,j) = 1;
       end
      end
      end
     end
    I2 = imbinarize(R)
    I3 = ~I2
  
    I4 = imfill(I3,'holes')
    I4 = bwareaopen(I4,30); %remove tiny objects

    [labeledImage, numShapes] = bwlabel(I4)
    number_of_shapes = numShapes %number of flakes

    a = regionprops(labeledImage, 'area')
    shapes_area = transpose(cell2mat(struct2cell(a))) % area of flakes

    b = regionprops(labeledImage, 'Solidity')
    shapes_solidity = transpose(cell2mat(struct2cell(b))) %solidity
    
    d = regionprops(labeledImage, 'perimeter')
    shapes_perimeter = transpose(cell2mat(struct2cell(d))) %perimeter

    e = regionprops(labeledImage, 'circularity')
    shapes_circularity = transpose(cell2mat(struct2cell(e))) %circularity

    f = regionprops(labeledImage, 'EquivDiameter')
    shapes_EquivDiameter = transpose(cell2mat(struct2cell(f))) %EquivDiameter
    
    k = regionprops(labeledImage, 'Eccentricity')
    shapes_Eccentricity = transpose(cell2mat(struct2cell(k))) %Eccentricity
    
    v = regionprops(labeledImage, 'MajorAxisLength')
    shapes_MajorAxisLength = transpose(cell2mat(struct2cell(v))) %MajorAxisLength
    
    p = regionprops(labeledImage, 'MinorAxisLength')
    shapes_MinorAxisLength = transpose(cell2mat(struct2cell(p))) %MinorAxisLength
    
    ratio_Equivdiameter_perimeter = (shapes_EquivDiameter)./shapes_perimeter
    ratio_majoraxislength_minoraxislength = shapes_MajorAxisLength./shapes_MinorAxisLength

    I5 = imbinarize(G)
    I6 = bwareaopen(I5,20)
    BW = imfill(I6, 'holes')

    [labeledImage, numberOfObjects] = bwlabel(BW)
    number_of_lightspot = numberOfObjects % number of lightspots
   
    shapes_solidity1 = [ mean(shapes_solidity), median(shapes_solidity),std(shapes_solidity),max(shapes_solidity),min(shapes_solidity),prctile(shapes_solidity,20),prctile(shapes_solidity,40),prctile(shapes_solidity,60),prctile(shapes_solidity,80)];
    shapes_perimeter1 = [ mean(shapes_perimeter), median(shapes_perimeter),std(shapes_perimeter),max(shapes_perimeter),min(shapes_perimeter),prctile(shapes_perimeter,20),prctile(shapes_perimeter,40),prctile(shapes_perimeter,60),prctile(shapes_perimeter,80)];
    shapes_circularity1 = [ mean(shapes_circularity), median(shapes_circularity),std(shapes_circularity),max(shapes_circularity),min(shapes_circularity),prctile(shapes_circularity,20),prctile(shapes_circularity,40),prctile(shapes_circularity,60),prctile(shapes_circularity,80)];
    shapes_Eccentricity1 = [ mean(shapes_Eccentricity), median(shapes_Eccentricity),std(shapes_Eccentricity),max(shapes_Eccentricity),min(shapes_Eccentricity),prctile(shapes_Eccentricity,20),prctile(shapes_Eccentricity,40),prctile(shapes_Eccentricity,60),prctile(shapes_Eccentricity,80)]; 
    ratio_Equivdiameter_perimeter1 = [ mean(ratio_Equivdiameter_perimeter), median(ratio_Equivdiameter_perimeter),std(ratio_Equivdiameter_perimeter),max(ratio_Equivdiameter_perimeter),min(ratio_Equivdiameter_perimeter),prctile(ratio_Equivdiameter_perimeter,20),prctile(ratio_Equivdiameter_perimeter,40),prctile(ratio_Equivdiameter_perimeter,60),prctile(ratio_Equivdiameter_perimeter,80)];
    ratio_majoraxislength_minoraxislength1 = [ mean(ratio_majoraxislength_minoraxislength), median(ratio_majoraxislength_minoraxislength),std(ratio_majoraxislength_minoraxislength),max(ratio_majoraxislength_minoraxislength),min(ratio_majoraxislength_minoraxislength),prctile(ratio_majoraxislength_minoraxislength,20),prctile(ratio_majoraxislength_minoraxislength,40),prctile(ratio_majoraxislength_minoraxislength,60),prctile(ratio_majoraxislength_minoraxislength,80)];
    ratio_num_lightspots_shapes1 = number_of_lightspot/number_of_shapes
    features = [number_of_shapes, shapes_solidity1,shapes_perimeter1,shapes_circularity1,shapes_Eccentricity1,number_of_lightspot,entropy_1, ratio_Equivdiameter_perimeter1,ratio_majoraxislength_minoraxislength1,ratio_num_lightspots_shapes1]

    trans_featureVector = transpose(features);
    TotalfeatureVector = [TotalfeatureVector,trans_featureVector];
end 
