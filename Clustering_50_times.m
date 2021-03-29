%% run the clustering algorithm 50 times
matching_rate_matrix = []
for i = 1: 50
   [matching_rate] =  Clustering_Func()
   matching_rate_matrix = [matching_rate_matrix, matching_rate]
end

boxplot(matching_rate_matrix)


