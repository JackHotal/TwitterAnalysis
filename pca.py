import numpy as np
from sklearn.decomposition import PCA

def pca_func(features):
  X = np.array(features)
  pca = PCA(n_components=2)
  pca.fit(X)
  print(pca)
  PCA(n_components=2)
  print(pca.explained_variance_ratio_)
  print(pca.singular_values_)

  pca = PCA(n_components=3, svd_solver='full')
  pca.fit(X)
  PCA(n_components=3, svd_solver='full')
  print(pca.explained_variance_ratio_)
  print(pca.singular_values_)

  pca = PCA(n_components=4, svd_solver='arpack')
  pca.fit(X)
  PCA(n_components=4, svd_solver='arpack')
  print(pca.explained_variance_ratio_)
  print(pca.singular_values_)

  return pca
  #return best fitting?!?!?!?