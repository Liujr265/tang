from sklearn import cluster
from sklearn import datasets
from sklearn import metrics
from sklearn import mixture
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
from time import time

def cal( algo,labels_true, labels_pred):
    print('%-30s\t%.3f\t%.3f\t%.3f' % (
          algo,
          metrics.normalized_mutual_info_score(labels_true, labels_pred,average_method='arithmetic'),
          metrics.homogeneity_score(labels_true, labels_pred),
          metrics.completeness_score(labels_true, labels_pred),
         ))
print('%-30s\t%s\t\t%s\t%s' % ('',  'Nmi', 'Homo', 'Comp'))
labels_pred = cluster.KMeans(n_clusters=7, random_state=11).fit_predict(scale(datasets.load_digits().data))
cal( 'K-Means',datasets.load_digits().target, labels_pred)
labels_pred = cluster.AffinityPropagation(damping=0.6, preference=-2600).fit_predict(scale(datasets.load_digits().data))
cal('AffinityPropagation',datasets.load_digits().target, labels_pred)
labels_pred = cluster.MeanShift(bandwidth=6).fit_predict(scale(datasets.load_digits().data))
cal( 'Mean-Shift',datasets.load_digits().target, labels_pred)
labels_pred = cluster.SpectralClustering(n_clusters=4, affinity='nearest_neighbors').fit_predict(scale(datasets.load_digits().data))
cal('SpectralClustering',datasets.load_digits().target, labels_pred)
labels_pred = cluster.AgglomerativeClustering(n_clusters=5).fit_predict(scale(datasets.load_digits().data))
cal('AgglomerativeClustering',datasets.load_digits().target, labels_pred)
labels_pred = cluster.DBSCAN(eps=1, min_samples=5).fit_predict(scale(datasets.load_digits().data))
cal( 'Dbscan',datasets.load_digits().target, labels_pred)
labels_pred = mixture.GaussianMixture(n_components=8).fit_predict(scale(datasets.load_digits().data))
cal('GaussianMixtures',datasets.load_digits().target, labels_pred)


