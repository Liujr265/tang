import numpy as np
from sklearn import cluster
from sklearn import datasets
from sklearn import metrics
from sklearn import mixture
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from time import time
cate = [
    'alt.atheism',
    'talk.religion.misc',
    'comp.graphics',
    'sci.space',
]
tar = datasets.fetch_20newsgroups(subset='all', categories=cate,
                                   shuffle=True, random_state=42).target
vectorizer = TfidfVectorizer(max_df=0.5, max_features=10000,
                                 min_df=2, stop_words='english',
                                 use_idf=True)
Data = make_pipeline(TruncatedSVD(2),Normalizer(copy=False)).fit_transform(vectorizer.fit_transform(datasets.fetch_20newsgroups(subset='all', categories=cate,
                                   shuffle=True, random_state=42).data))
labels=datasets.fetch_20newsgroups(subset='all', categories=cate,
                                   shuffle=True, random_state=42).target
print('%-30s\t%s\t\t%s\t%s' % ('',  'Nmi', 'Homo', 'Comp'))
def cal(algo,labels_true, labels_pred):
    print('%-30s\t%.3f\t%.3f\t%.3f' % (
          algo,
          metrics.normalized_mutual_info_score(labels_true, labels_pred,average_method='arithmetic'),
          metrics.homogeneity_score(labels_true, labels_pred),
          metrics.completeness_score(labels_true, labels_pred),
         ))
labels_pred = cluster.KMeans(n_clusters= np.unique(tar).shape[0], random_state=30).fit_predict(Data)
cal( 'K-Means',labels, labels_pred)
labels_pred = cluster.AffinityPropagation(damping=0.6, preference=-2000).fit_predict(Data)
cal('AffinityPropagation',labels, labels_pred)
labels_pred = cluster.MeanShift(bandwidth=0.0005, bin_seeding=True).fit_predict(Data)
cal('Mean-Shift',labels, labels_pred)
labels_pred = cluster.SpectralClustering(n_clusters= np.unique(tar).shape[0]).fit_predict(Data)
cal('SpectralClustering',labels, labels_pred)
labels_pred = cluster.AgglomerativeClustering(n_clusters= np.unique(tar).shape[0]).fit_predict(Data)
cal('AgglomerativeClustering',labels, labels_pred)
labels_pred = cluster.DBSCAN(eps=0.004, min_samples=6).fit_predict(Data)
cal('Dbscan',labels, labels_pred)
labels_pred = mixture.GaussianMixture(n_components= np.unique(tar).shape[0]).fit_predict(Data)
cal('GaussianMixtures',labels, labels_pred)
