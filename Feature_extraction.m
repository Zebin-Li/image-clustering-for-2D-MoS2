%% extract features from original optical images
imds = imageDatastore(fullfile('E:\',{'new images 4.4.2020'}), ...
'LabelSource','foldernames','FileExtensions', {'.tif'});
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
       if G(i,j) < 206  %manually tuned for lightspots
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

    b = regionprops(labeledImage, 'Solidity')
    shapes_solidity = transpose(cell2mat(struct2cell(b))) %solidity
    SS = shapes_solidity
    
    d = regionprops(labeledImage, 'perimeter')
    shapes_perimeter = transpose(cell2mat(struct2cell(d))) %perimeter
    SP = shapes_perimeter

    e = regionprops(labeledImage, 'circularity')
    shapes_circularity = transpose(cell2mat(struct2cell(e))) %circularity
    SC = shapes_circularity

    f = regionprops(labeledImage, 'EquivDiameter')
    shapes_EquivDiameter = transpose(cell2mat(struct2cell(f))) %EquivDiameter
    SE = shapes_EquivDiameter
    
    k = regionprops(labeledImage, 'Eccentricity')
    shapes_Eccentricity = transpose(cell2mat(struct2cell(k))) %Eccentricity
    SEC = shapes_Eccentricity
    
    v = regionprops(labeledImage, 'MajorAxisLength')
    shapes_MajorAxisLength = transpose(cell2mat(struct2cell(v))) %MajorAxisLength
    
    p = regionprops(labeledImage, 'MinorAxisLength')
    shapes_MinorAxisLength = transpose(cell2mat(struct2cell(p))) %MinorAxisLength
    
    ratio_Equivdiameter_perimeter = (shapes_EquivDiameter)./shapes_perimeter
    REP = ratio_Equivdiameter_perimeter
    
    ratio_majoraxislength_minoraxislength = shapes_MajorAxisLength./shapes_MinorAxisLength
    RMM = ratio_majoraxislength_minoraxislength

    I5 = imbinarize(G)
    I6 = bwareaopen(I5,20)
    BW = imfill(I6, 'holes')

    [labeledImage, numberOfObjects] = bwlabel(BW)
    number_of_lightspot = numberOfObjects % number of lightspots
   
    shapes_solidity1 = [ mean(SS), median(SS),std(SS),max(SS),min(SS), ...
        prctile(SS,20),prctile(SS,40),prctile(SS,60),prctile(SS,80)];
    shapes_perimeter1 = [ mean(SP), median(SP),std(SP),max(SP),min(SP), ...
        prctile(SP,20),prctile(SP,40),prctile(SP,60),prctile(SP,80)];
    shapes_circularity1 = [ mean(SC), median(SC),std(SC),max(SC),min(SC), ...
        prctile(SC,20),prctile(SC,40),prctile(SC,60),prctile(SC,80)];
    shapes_Eccentricity1 = [ mean(SE), median(SE),std(SE),max(SE),min(SE), ...
        prctile(SE,20),prctile(SE,40),prctile(SE,60),prctile(SE,80)]; 
    ratio_Equivdiameter_perimeter1 = [ mean(REP), median(REP),std(REP), ...
        max(REP),min(REP),prctile(REP,20),prctile(REP,40),prctile(REP,60),prctile(REP,80)];
    ratio_majoraxislength_minoraxislength1 = [ mean(RMM), median(RMM), ...
        std(RMM),max(RMM),min(RMM),prctile(RMM,20),prctile(RMM,40), ...
        prctile(RMM,60),prctile(RMM,80)];
    ratio_num_lightspots_shapes1 = number_of_lightspot/number_of_shapes
    
    features = [number_of_shapes, shapes_solidity1,shapes_perimeter1,shapes_circularity1, ...
        shapes_Eccentricity1,number_of_lightspot,entropy_1, ratio_Equivdiameter_perimeter1, ...
        ratio_majoraxislength_minoraxislength1,ratio_num_lightspots_shapes1]

    trans_featureVector = transpose(features);
    TotalfeatureVector = [TotalfeatureVector,trans_featureVector];
end 
