class ClusteringUtil:
    
    def plot_avg_silhouette_score(X, range_n_clusters):
        """
        plot average silhouette score against a range of values of n_clusters

        Parameters
        ----------
        X : np.ndarray
            feature matrix
        range_n_clusters : np.ndarray
            a range of values for n_clusters

        Returns
        -------
        """
        silhouette_avgs = []

        for n_clusters in tqdm(range_n_clusters):
            #n_init is increased from default value 10 to alleviate sensitivity of KMean cluster wrt initial centroid
            model = KMeans(n_clusters = n_clusters, n_init = 20) 
            cluster_labels = model.fit_predict(X)
            silhouette_avg = silhouette_score(X, cluster_labels)
            silhouette_avgs.append(silhouette_avg)

        plt.figure(figsize = (8,6))
        plt.plot(range_n_clusters, silhouette_avgs)
        plt.xlabel("k")
        plt.ylabel("avg silhouette score")
        plt.title("KMean")
        plt.show()
    
    
    def plot_clusters(data, cluster_assignment_col, features_to_plot):
        """
        plot clusters for visualization

        Parameters
        ----------
        data : pd.DataFrame
            at merchant level with cluster assignment
        cluster_assignment_col : str
            name of the cluster assignment column
        features_to_plot : list
            names of features to plot the first two dimensions
        Returns
        -------
        """
        cluster_assignment = data[cluster_assignment_col]
        clusters = np.unique(cluster_assignment)

        print(f"number of clusters: {len(clusters)}")
        for cluster in clusters:
            row_ix = np.where(cluster_assignment == cluster)
            f0, f1 = features_to_plot[0], features_to_plot[1]
            plt.scatter(data.loc[row_ix, f0], data.loc[row_ix, f1], label=str(cluster))
            plt.xlabel(f0)
            plt.ylabel(f1)
        plt.legend()
        plt.title("KMean")
        plt.show()
