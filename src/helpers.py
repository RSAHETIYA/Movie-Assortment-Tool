import numpy as np
from matplotlib import pyplot as plt
from sklearn import datasets
from sklearn.feature_selection import VarianceThreshold
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score, explained_variance_score, mean_squared_error
from sklearn import linear_model, svm
from sklearn.kernel_ridge import KernelRidge
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
from sklearn.svm import OneClassSVM
from sklearn.cluster import DBSCAN
import seaborn as sn
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn import metrics
from scipy.spatial.distance import cdist

# Load the Pandas libraries with alias 'pd' 
import pandas as pd 

rawRatings = pd.read_csv("ratings.csv") 
rawTagNames = pd.read_csv("genome-tags.csv")
rawTagRel = pd.read_csv("genome-scores.csv")
rawMovTitl = pd.read_csv("movies.csv")
copyofRawMovTitl = pd.DataFrame(rawMovTitl)

# Cast to numpy arrays
rawRatings = np.array(rawRatings)
rawTagRel = np.array(rawTagRel)
rawMovTitl = np.array(rawMovTitl)

# Debug
def checkForBadEntries(matrix, column, listPrintout = False):
  print("Contains finite entries in col ", column, ": " , np.isfinite(matrix[:, column]).all())
  containNan = np.isnan(matrix[:, column]).any()
  print("Contains NAN in col ", column, ": ", containNan)
  if (containNan and listPrintout):
    boolList = np.isnan(matrix[:, 2])
    for i in range(len(boolList)):
      if boolList[i] == True:
        print(i)
        
  allNot = True
  for i in range(len(matrix)):
    if matrix[i, column] == 0:
      allNot = False
      if (listPrintout):
        print(i)
  print("Contains no zero entries: ", allNot)

  allNot = True
  for i in range(len(matrix)):
    if matrix[i, column] == -1:
      allNot = False
      if (listPrintout):
        print(i)
  print("Contains no -1 entries: ", allNot)

def removeUnused(matrix, col, val):
  return np.array(matrix[matrix[:, col] != val, :])

def generateDesign():
  # Fill with negatives ones; when finished, negatives will all be removed
  design = np.ones((rawMovTitl[rawMovTitl.shape[0] - 1, 0], 3))
  design = np.negative(design)

  # Movie ID, Title
  for row in rawMovTitl:
    design[row[0] - 1, :] = np.append(row[0], np.zeros((2)))
  print(design[0])

  # Total Rating
  for row in rawRatings:
    index = int(row[1] - 1)
    rating = row[2]
    design[index, 1] = design[index, 1] + 1
    design[index, 2] = design[index, 2] + rating

  # Resize matrix in preparation for addition of tag relevance scores
  num_tags = rawTagNames.shape[0]
  negOnes = np.negative(np.ones((design.shape[0], num_tags)))
  design = np.append(design, negOnes, axis = 1)
  
  # Put feature names in a separate array
  #designColNames = np.chararray((design.shape[1]))
  #designColNames[0] = "Movie ID"
  #designColNames[1] = "Number of Ratings"
  #designColNames[2] = "Average Star Rating"
  #for i in range(rawTagNames.shape[0]):
  #  designColNames[rawTagNames[i, 0] + 2] = rawTagNames[i, 1]
  #print("Put feature names into separate array")

  # Tag relevances
  i = 0 
  while i < rawTagRel.shape[0]:
    id = int(rawTagRel[i, 0])
    tagRelVector = rawTagRel[i:(i + num_tags), 2] # [inclusive:exclusive]
    design[id - 1, :] = np.append(design[id - 1, 0:3], tagRelVector, axis=0)
    i += num_tags # Because the for loop itself already adds 1
  print("Appended tag vectors")
  print(design.shape)
  print(design[0, 0:5])

  # Removes unused entries in the matrix and unrated movies. This accounts for
  #   both ids with no movie as well as movies without ratings or tag
  #   relevances. Column 4 is an arbitrary choice out of columns [3, 1131]
  design = removeUnused(design, 4, -1)
  design = removeUnused(design, 1, 0)
  print("Removed empty")

  # Convert sum of ratings to average rating
  design[:, 2] = design[:, 2] / design[:, 1]

  return design

doNotModifyDesign = generateDesign()

# Returns a DEEP COPY of the design matrix for your use 
def getDesign():
  return np.array(doNotModifyDesign)
  
# Feature selection based on variance threshold
def runElimLowVarFeats(feats, thresh = .06):
  local = np.array(feats)
  vThresh = VarianceThreshold(threshold=(thresh))
  local = vThresh.fit_transform(local)
  print(feats.shape, "to" , local.shape)
  return local

# PCA
#   Note componentFactor may be a ratio or an actual amount of components
def runPCA(feats, componentFactor=float(.95)):
  local = np.array(feats)
  pca = PCA(n_components=componentFactor)
  local = pca.fit_transform(local)
  print(feats.shape, "to" , local.shape)
  return local

# TBA: Forward Selection?
# https://planspace.org/20150423-forward_selection_with_statsmodels/ 

# ----- Unsupervised Learning-Based -----

# Sort by a given column
def sortByCol(matrix, col):
  matrix = matrix[matrix[:, col].argsort()] 
  matrix = np.flip(matrix, axis = 0)
  return matrix

# Sorts and finds the top. Default value is set here to 500 movies
def generateMostPopular(cap=500):
  local = getDesign()
  local = sortByCol(local, 1)
  return local[0:cap]

# Remove popularity, avg rating
def simplifyUnsupFeats(matrix):
  local = np.array(matrix)
  local = np.append(matrix[:, 0:1], matrix[:, 3:], axis = 1)
  return local

# Return only the feature space, excludes Movie ID.
def separate(matrix):
  local = np.array(matrix)
  return local[:, 0], local[:, 1:]

# Give names to Movie IDs, pair with respective assignments
def interpretClusters(clusterAssigns, movIDs):
  together = []
  for i in range(movIDs.shape[0]):
    title = ""
    for movie in copyofRawMovTitl.itertuples():
      if (int(movIDs[i]) == int(movie[1])):
        title = movie[2]
        #print(title)
        break
    together.append([clusterAssigns[i], title])
  together.sort(key = lambda together: together[0])
  return together

# Prints out the actual movie names and the amount of movies in each cluster
def displayPairs(twoDList):
  previousCluster = -1
  numInEach = []
  for mov in twoDList:
    if (mov[0] != previousCluster):
      print("--------- NEW CLUSTER -----------")
      previousCluster = mov[0]
      numInEach.append(1)
    else:
      numInEach[previousCluster] += 1
    print(mov)
  print(numInEach)

# ----- Supervised Learning-Based -----

# Splits into testing and training sets, each with labels and features
#   Label is popularity
def partition(matrix, trainingP = .25):
  mTrain, mTest = train_test_split(matrix, train_size = trainingP, random_state=4)
  trainLabel = mTrain[:, 1]
  trainFeats = mTrain[:, 3:]
  testLabel = mTest[:, 1]
  testFeats = mTest[:, 3:]
  return trainFeats, trainLabel, testFeats, testLabel
# Removes from the top end
def removeOutliers(matrix, thresh = 62500):
  local = np.array(matrix)
  local = local[local[:, 1] <= thresh, :]
  return local
# Combine training and 

# ----- Normalization -----
# Take log of a feature given a column. By default, this value is set to the
#   popularity column. 
def logFeat(matrix, col=1):
  local = np.array(matrix)
  local[:, col] = np.log(local[:, col])
  return local
# Divide by max in feature / column. By default, set to the Avg Star Rat. column
def divByMax(matrix, col=2):
  local = np.array(matrix)
  local[:, col] = local[:, col] / local[:, col].max()
  return local

# ----- Presets -----
# Design matrix with columns {id, popularity, avg ratings, tag rels ... }
def supervisedPreset():
  design = getDesign()
  design = removeOutliers(design)
  design = logFeat(design, 1)
  design = divByMax(design, 1)
  return design
  
# Design matrix with columns {id, avg ratings, tag rels ... }
def unsupPreset(cap = 500):
  design = generateMostPopular(cap)
  design = simplifyUnsupFeats(design)
  return design
  
def plotSeparatePCA(test_feats, test_labels, pred):
  
  screePCA = PCA(8)
  screePCA.fit(test_feats)
  for pcaComponent in screePCA.components_:
    pcaComponent = np.sort(np.array(pcaComponent))
    print(pcaComponent.shape)
    test_labels = np.sort(np.array(test_labels[0:pcaComponent.shape[0]]).flatten())
    print(test_labels.shape)
    pred = np.sort(np.array(pred[0:pcaComponent.shape[0]]).flatten())
    print(pred.shape)

    plt.scatter(pcaComponent, test_labels, color='black', s = 1)
    plt.plot(pcaComponent, pred, color='blue', linewidth=3)
    plt.title('Impace of PCA Component on Popularity Using Neural Net Model')
    plt.xlabel('Principal Component')
    plt.ylabel('Popularity')
    plt.show()

#def graph(x_range, formula, otherx, othery):
#   x = np.array(x_range)
#   y = eval(formula)
#   plt.plot(x, y)
#   plt.scatter(othery, otherx)

# Preliminary metrics
def regMetrics(ground, pred):

  plt.show()
  rSq = r2_score(ground, pred)
  explainedV = explained_variance_score(ground, pred)
  rmse = np.sqrt(mean_squared_error(ground, pred))
  print("R^2:", rSq)
  print("Explained Variance:", explainedV)
  print("RMSE:", rmse)
  return rSq, explainedV, rmse
  # https://scikit-learn.org/stable/modules/model_evaluation.html 

def plotReg(test_feats, test_labels, pred):
  oneFeat = runPCA(test_feats, 1)
  oneFeat = np.resize(oneFeat, (oneFeat.shape[0]))
  oneFeat = np.sort(oneFeat)
  test_feats = np.sort(np.array(test_feats))
  pred = np.sort(np.array(pred))
  plt.scatter(oneFeat, test_labels, color='black', s = 1)
  plt.plot(oneFeat, pred, color='blue', linewidth=3)
  print(test_feats.shape, oneFeat.shape, test_labels.shape, pred.shape)
  plt.show()

# Linear Regression
def linReg(train_feats, train_labels, test_feats, test_labels):
  regr_lin = linear_model.LinearRegression()
  regr_lin.fit(train_feats, train_labels)
  pred_lin = regr_lin.predict(test_feats)
  plotReg(test_feats, test_labels, pred_lin)
  print("\nLin Reg Results: ")
  rSq, explainedV, rmse = regMetrics(test_labels, pred_lin)

# Questionable use of SVM regression
def svmReg(train_feats, train_labels, test_feats, test_labels):
  regr_svm = svm.SVR()
  regr_svm.fit(train_feats, train_labels)
  pred_svm = regr_svm.predict(test_feats)
  plotReg(test_feats, test_labels, pred_svm)
  print("\nSVM Reg Results: ")
  rSq, explainedV, rmse = regMetrics(test_labels, pred_svm)

# Lasso Regression
def lassoReg(train_feats, train_labels, test_feats, test_labels):
  regr_lasso = linear_model.Lasso()
  regr_lasso.fit(train_feats, train_labels)
  pred_lasso = regr_lasso.predict(test_feats)
  plotReg(test_feats, test_labels, pred_lasso)
  print("\nLasso Reg Results: ")
  rSq, explainedV, rmse = regMetrics(test_labels, pred_lasso)

# Linear Ridge Regression
def linRidgeReg(train_feats, train_labels, test_feats, test_labels):
  regr_lin = linear_model.Ridge()
  regr_lin.fit(train_feats, train_labels)
  pred_lin = regr_lin.predict(test_feats)
  plotReg(test_feats, test_labels, pred_lin)
  print("\nLin Ridge Reg Results: ")
  rSq, explainedV, rmse = regMetrics(test_labels, pred_lin)

# Kernel Ridge Regression
def kernelRidge(train_feats, train_labels, test_feats, test_labels, inputKern = "linear"):
  ker = KernelRidge(kernel = inputKern)
  ker.fit(train_feats, train_labels)
  pred = ker.predict(test_feats)
  plotReg(test_feats, test_labels, pred)
  print("\nKernel Ridge Reg Results: \n\t (with", inputKern, "kernel type:)")
  rSq, explainedV, rmse = regMetrics(test_labels, pred)


# implement later
def anomalyDetect(x):
  #x = runPCA(x, 2)
  plt.scatter(x[:,0], x[:,1])
  plt.show()

  svm = OneClassSVM(kernel='rbf', gamma=0.001, nu=0.02)
  print(svm)

  pred = svm.fit_predict(x)
  scores = svm.score_samples(x)

  thresh = np.quantile(scores, 0.99)
  print(thresh)
  index = np.where(scores>=thresh)
  values = x[index]

  plt.scatter(x[:,0], x[:,1])
  plt.scatter(values[:,0], values[:,1], color='r')
  plt.xlabel("PCA Component 1")
  plt.ylabel("PCA Component 2")
  plt.show()

def testKernel(X):
  # Removed: "chi2", 
  # additive chi2 because it took a long time to run
  # chi2 because it broke
  kerns = ["additive_chi2", "linear", "poly", "polynomial", "rbf", "laplacian", "sigmoid", "cosine"]
  for kern in kerns:
    trainFeats, trainLabel, testFeats, testLabel = partition(X)
    kernelRidge(trainFeats, trainLabel, testFeats, testLabel, kern)

def supKitchenSink(X):
  X = np.array(X)
  if (X.shape[1] <= 3):
    print("Incorrect feature dimensions")
    return

  # Linear Regression
  trainFeats, trainLabel, testFeats, testLabel = partition(X)
  linReg(trainFeats, trainLabel, testFeats, testLabel)

  # Linear Ridge Regression
  trainFeats, trainLabel, testFeats, testLabel = partition(X)
  linRidgeReg(trainFeats, trainLabel, testFeats, testLabel)

  # SVM Regression
  trainFeats, trainLabel, testFeats, testLabel = partition(X)
  svmReg(trainFeats, trainLabel, testFeats, testLabel)

  # Linear Lasso Regression
  trainFeats, trainLabel, testFeats, testLabel = partition(X)
  lassoReg(trainFeats, trainLabel, testFeats, testLabel)

  # All types of Kernel Ridge Regression
  testKernel(X)

def runPCAexclLabel(matrix, N=.95):
  local = np.array(matrix)
  local = np.append(local[:, 0:3], runPCA(local[:, 3:], componentFactor=N), axis = 1)
  return local

def runElimLowVarFeatsExclLabel(matrix, t=.06):
  local = np.array(matrix)
  local = np.append(local[:, 0:3], runElimLowVarFeats(local[:, 3:], thresh = t), axis = 1)
  return local
  

def kMeansClustering(featureSet, num_clusters = 8):
  kModel = KMeans(num_clusters)
  kModel.fit(featureSet)
  # Metrics
  return kModel.labels_

# GMM model

def GMMClustering(featureSet, n = 8, covType = "full"):
    gmm = GaussianMixture(n_components = n, covariance_type = covType)
    return gmm.fit_predict(featureSet)
    


def runDB(matrix, eps = 5, min_samples = 4):
  db = DBSCAN(eps, min_samples).fit(matrix)
  core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
  core_samples_mask[db.core_sample_indices_] = True
  labels = db.labels_
  labelsToReturn = np.array(labels)

  # Number of clusters in labels, ignoring noise if present.
  n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
  n_noise_ = list(labels).count(-1)

  print('Estimated number of clusters: %d' % n_clusters_)
  print('Estimated number of noise points: %d' % n_noise_)

  # Deprecated Graphing
  # Black removed and is used for noise instead.
  '''
  unique_labels = set(labels)
  colors = [plt.cm.Spectral(each)
            for each in np.linspace(0, 1, len(unique_labels))]
  for k, col in zip(unique_labels, colors):
      if k == -1:
          # Black used for noise.
          col = [0, 0, 0, 1]
      class_member_mask = (labels == k)
      xy = matrix[class_member_mask & core_samples_mask]
      plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
              markeredgecolor='k', markersize=14)
      xy = matrix[class_member_mask & ~core_samples_mask]
      plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
              markeredgecolor='k', markersize=6)
  plt.title('Estimated number of clusters: %d' % n_clusters_)
  plt.show()
  '''

  return labelsToReturn
  
# Unsupervised Graph
def graphUnsup(features, labels):
  reducedFeatures = runPCA(features, 2)
  pca_data = np.transpose(np.vstack((np.transpose(reducedFeatures), labels)))
  pca_df = pd.DataFrame(data = pca_data, columns=("Principal_Component_1", "Principal_Component_2", "Label"))
  sn.FacetGrid(pca_df, hue="Label", size = 6).map(plt.scatter, "Principal_Component_1", "Principal_Component_2").add_legend()
  plt.show()
  
# Clustering metrics
def unsupMetrics(feats, labels):
  allSame = True
  label0 = labels[0]
  for entry in labels:
    if entry != label0:
      allSame = False
  if (allSame):
    return 0, 5, 0
  sil = silhouette_score(feats, labels)
  db = davies_bouldin_score(feats, labels)
  ch = calinski_harabasz_score(feats, labels)
  print("Silhouette Coefficient: ", sil)
  print("Davies-Bouldin Index: ", db)
  print("Calinski-Harabasz Score: ", ch)
  return sil, db, ch




def numKElbowGen(feats, upperLim=20):
  X = np.array(feats)
  distortions = []
  inertias = []

  mapping1 = {}
  mapping2 = {}

  K = range(1, upperLim)
  for k in K:
    # Building and fitting the model
    kmeanModel = KMeans(n_clusters=k).fit(X)
    kmeanModel.fit(X)
    distortions.append(sum(np.min(cdist(X, kmeanModel.cluster_centers_,
                                        'euclidean'), axis=1)) / X.shape[0])
    inertias.append(kmeanModel.inertia_)
    mapping1[k] = sum(np.min(cdist(X, kmeanModel.cluster_centers_,
                                    'euclidean'), axis=1)) / X.shape[0]
    mapping2[k] = kmeanModel.inertia_

  plt.plot(K, distortions, 'bx-')
  plt.xlabel('Values of K')
  plt.ylabel('Distortion')
  plt.title('The Elbow Method using Distortion')
  plt.show()

  plt.plot(K, inertias, 'bx-')
  plt.xlabel('Values of K')
  plt.ylabel('Inertia')
  plt.title('The Elbow Method using Inertia')
  plt.show()

def unsupKitchenSink(feats, ids, cStart = 6, cNonincEnd = 13, dbStart = 3.5, dbStep=.1, dbNumIter = 10, dbN = 5, testNumInit = 0):
  
  # Test types: 0 indicates Kmeans
  #             1 GMM
  #             2 DBSCAN
  results = np.zeros((int(2 * (cNonincEnd - cStart) + dbNumIter), 5))
  testCounter = testNumInit
  
  print("ELBOW GRAPH: ")
  numKElbowGen(feats)
  
  rang = range(cStart, cNonincEnd)
  print("Running regular KMeans on range")
  for i in rang:
    print("\n", "Test", testCounter,
          "\nTest type: KMeans",
          "\nParameters:", "Num Clusters i =", i)
    labels = kMeansClustering(feats, i)
    displayPairs(interpretClusters(labels, ids))
    graphUnsup(feats, labels)
    sil, db, ch = unsupMetrics(feats, labels)
    results[testCounter - testNumInit, :] = np.array([testCounter, 0, sil, db, ch])
    testCounter += 1
  
  print("Running regular GMM on range")
  for i in rang:
    print("\n", "Test", testCounter,
          "\nTest type: GMM",
          "\nParameters:", "Num Components i =", i)
    labels = GMMClustering(feats, i)
    displayPairs(interpretClusters(labels, ids))
    graphUnsup(feats, labels)
    sil, db, ch = unsupMetrics(feats, labels)
    results[testCounter - testNumInit, :] = np.array([testCounter, 1, sil, db, ch])
    testCounter += 1
  
  dbRang = np.arange(dbStart, dbStep * dbNumIter + dbStart, dbStep)
  for l in dbRang:
    print("\n", "Test", testCounter,
          "\nTest type: DBScan",
          "\nParameters:", "Radius epsilon =", l, "Min samples n =", dbN)
    labels = runDB(feats, l, dbN)
    print(labels.shape, ids.shape)
    print(labels[0], ids[0])
    for k in labels:
      if (k == -1):
        labels = labels + 1
        break
    displayPairs(interpretClusters(labels, ids))
    graphUnsup(feats, labels)
    sil, db, ch = unsupMetrics(feats, labels)
    results[testCounter - testNumInit, :] = np.array([testCounter, 2, sil, db, ch])
    testCounter += 1

  return results
  

