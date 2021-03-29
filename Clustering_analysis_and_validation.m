% Clustering Analysis and Validation
close all;
clear;
clc;

%% Custering Analysis
%%load data
load rating_label_matching_new.mat
load TF6.mat

%% data normalization
x = normalize([TF6(:,1:37),TF6(:,47:67)])
x1 = transpose(x)

%% PCA
[x1,pca_settings] = processpca(x1,0.015); 

%% SOM
b = 15  
net = selforgmap([b b]);
net.layers{1}.transferFcn = 'tansig';
[net,tr] = train(net,x1);
y = net(x1);  
cluster_index = vec2ind(y);
x1_trans = transpose(x1)
neuron_centroid = net.IW    
n = cell2mat(neuron_centroid)
sample_hit_table = tabulate(cluster_index)

%% calculate dissimilarity matrix
dissimilarity_mtx = []
distance_neu_data = []

nn = length(n(:,1))
jj = length(x1_trans(:,1))
for i=1:nn
    for j=1:jj
    distance = norm(n(i,:,:,:,:,:,:,:,:,:,:,:,:,:) - ...
        x1_trans(j,:,:,:,:,:,:,:,:,:,:,:,:,:));
    distance_neu_data = [distance_neu_data,distance];
    end
    dissimilarity_mtx = [dissimilarity_mtx; distance_neu_data];
    distance_neu_data = []
end

%% k-means clustering
k = 2;
X = dissimilarity_mtx; 
idx = kmeans(X,k);

%% add the SOM label and K means label to the data matrix
x1_trans_1 = [x1_trans, transpose(cluster_index),zeros(length(x1_trans),1)]
for q = 1:529
    for w = 1:length(n)
    if x1_trans_1(q,10) == w
        x1_trans_1(q,11) = idx(w)  
    end
    end
end

%% visualization
figure
for r = 1:529
    if x1_trans_1(r,11) == 2  
        scatter(x1_trans_1(r,1),x1_trans_1(r,2),'b','+'); 
        hold on;
    elseif x1_trans_1(r,11) == 1  
                scatter(x1_trans_1(r,1),x1_trans_1(r,2),'g');
                hold on;
    end
end

%% Validation 
%%calculate matching rate
matching_count1 = 0
matching_count2 = 0
matching_rate_final = 0
matching_rate1 = 0
matching_rate2 = 0

for t = 1: 529
    if rating_label_matching_new(t) == 11 & x1_trans_1(t,11) == 1
        matching_count1 = matching_count1 + 1
    elseif rating_label_matching_new(t) == 22 & x1_trans_1(t,11) == 2
        matching_count1 = matching_count1 + 1
    end
end
matching_rate1 = matching_count1/529

for t = 1: 529
    if rating_label_matching_new(t) == 11 & x1_trans_1(t,11) == 2
        matching_count2 = matching_count2 + 1
    elseif rating_label_matching_new(t) == 22 & x1_trans_1(t,11) == 1
        matching_count2 = matching_count2 + 1
    end
end
matching_rate2 = matching_count2/529

%% decide the final matching rate
matching = [matching_rate1,matching_rate2]
matching_rate_final = max(matching)

%% draw confusion matrix
CX = string(rating_label_matching_new)
for i = 1: 529
    if rating_label_matching_new(i) == 11 %11 is bad 
        CX(i) = 'bad'
    elseif rating_label_matching_new(i) == 22 %11 is good
        CX(i) = 'good'
    end
end
CXX = string(x1_trans_1(:,11))
for i = 1:529
    if x1_trans_1(i,11) == 1
        CXX(i) = 'good'         %change here 
    elseif x1_trans_1(i,11)== 2
        CXX(i) = 'bad'
    end
end

figure;hold off;
cm = confusionchart(CX,CXX);
cm.Title = 'Validation Confusion Matrix';
cm.RowSummary = 'row-normalized';
cm.ColumnSummary = 'column-normalized';

