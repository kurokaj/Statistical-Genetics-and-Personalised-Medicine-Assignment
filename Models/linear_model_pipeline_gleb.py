import os
import numpy as np
import tensorflow as tf
from tensorflow import TensorSpec as ts
import pandas as pd
from scipy.special import logit
from matplotlib import pyplot as plt 
from tqdm import tqdm, trange
from time import time
from evaluate_sepsis_score import compute_prediction_utility

startTime = time()
dataDir = "../data"
dtype = np.double
reloadDataFlag = False
cvFoldN, cvFold = 5, 0
batchN = 10

if reloadDataFlag:
  subDirList = ["original/training_A", "original/training_B"]
  dataFrames, P = [], 0
  for subDir in subDirList:
    files = [f for f in os.listdir(os.path.join(dataDir,subDir)) if os.path.isfile(os.path.join(dataDir,subDir, f))]
    for f in tqdm(files):
      df = pd.read_csv(os.path.join(dataDir,subDir,f),sep="|")
      df["Id"] = P
      dataFrames.append(df)
      P += 1
  dataFrame = pd.concat(dataFrames)
  
  for i in range(34):
    q = np.nanquantile(dataFrame.iloc[:,i],[0,0.001,0.01,0.05,0.25,0.5,0.75,0.95,0.99,0.999,1])
    dataFrame.iloc[:,i] = np.clip(dataFrame.iloc[:,i], q[1], q[-2])
    plt.hist(dataFrame.iloc[:,i])
    plt.title(str(i) + " " + dataFrame.columns[i])
    plt.show()
    print(q)
  dataFrame.to_pickle(os.path.join(dataDir,"all_raw_data.pkl"))
else:
  dataFrame = pd.read_pickle(os.path.join(dataDir,"all_raw_data.pkl"))
  P = np.unique(dataFrame["Id"]).shape[0]
  
dataFrame["Unit"] = dataFrame["Unit1"] + 2*dataFrame["Unit2"] - 1 
dataFrame.loc[np.isnan(dataFrame["Unit"]),"Unit"] = 2
dataFrame["HospAdmTime"][np.isnan(dataFrame["HospAdmTime"])] = np.nanmean(dataFrame["HospAdmTime"])
dataFrameList = [df for _, df in dataFrame.groupby("Id")]

for p in tqdm(range(P)):
  df = dataFrameList[p]
  ind = np.where(df["SepsisLabel"]==1)[0]
  if ind.size > 0:
    indMin = ind[0]
    df = df.assign(Time=np.maximum(df["ICULOS"] - df.loc[df.index[indMin],"ICULOS"],-90), Deseased=1)
  else:
    df = df.assign(Time=np.nan, Deseased=0)
  dataFrameList[p] = df
dataFrame = pd.concat(dataFrameList)


YColumns = dataFrame.columns[:34]
XColumns = ["Time", "Deseased", "Age", "Gender", "Unit", 'HospAdmTime', 'ICULOS', "Id"]
timeColumnIndex = 6
YAll = dataFrame[YColumns].values
XAll = dataFrame[XColumns].values
YScale = np.array([np.nanmean(YAll,0),np.nanstd(YAll,0)])
XScale = np.array([np.nanmean(XAll,0),np.nanstd(XAll,0)])
XScale[:,-1] = [0,1]
XScale[0,0] = 0
YAll = (YAll - YScale[0]) / YScale[1]
XAll = (XAll - XScale[0]) / XScale[1]

# 80-20 train-test patient-wize split
idVecTest = np.where((np.arange(P) % cvFoldN) == cvFold)[0]
idVecTrain = np.setdiff1d(np.arange(P), idVecTest)
indTrain = np.isin(XAll[:,-1], idVecTrain)
indTest = np.isin(XAll[:,-1], idVecTest)
XTrain, YTrain = XAll[indTrain], YAll[indTrain]
XTest, YTest = XAll[indTest], YAll[indTest]

XTrain = XTrain[:,:-1]
XTrain[np.isnan(XTrain)] = 0
YoTrain = (~np.isnan(YTrain)).astype(dtype)
YTrain[np.isnan(YTrain)] = 0
N,J = YTrain.shape
C = XTrain.shape[1]

# comment next line if you prefer eager execution mode
@tf.function(input_signature=(ts([C,J],dtype),ts([J],dtype),ts([None,C],dtype),ts([None,J],dtype),ts([None,J],dtype)))
def logLikelihood(B,sigma2, X,Y,Yo):
  Mu = tf.matmul(X, B)
  R = (Y - Mu) * Yo
  logDetD_vec = tf.reduce_sum(Yo * tf.math.log(sigma2), 1)
  qF_vec = tf.reduce_sum(tf.square(R) / sigma2, 1)
  constTerm_vec = -0.5*tf.reduce_sum(Yo, 1)*np.log(2*np.pi)
  logLike_vec = -0.5*(logDetD_vec + qF_vec) + constTerm_vec
  logLike = tf.reduce_sum(logLike_vec)
  return logLike, logLike_vec

B = tf.Variable(tf.zeros([C,J], dtype=dtype))
sigma2_uncon = tf.Variable(tf.zeros([J], dtype=dtype))
par_list = [B, sigma2_uncon]
optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)

# comment next line if you prefer eager execution mode
@tf.function(input_signature=(ts([None,C],dtype),ts([None,J],dtype),ts([None,J],dtype)))
def optimizer_step(X,Y,Yo):
  with tf.GradientTape() as tape:
    tape.watch(par_list)
    sigma2 = tf.math.softplus(sigma2_uncon)
    loss = -logLikelihood(B,sigma2, X,Y,Yo)[0] * (N/tf.shape(X)[0])
  grad_list = tape.gradient(loss, par_list)
  optimizer.apply_gradients(zip(grad_list,par_list))
  return loss

with trange(100) as tran:
  for epoch in tran:
    ind = np.random.permutation(np.arange(YTrain.shape[0]))
    for batch in range(batchN):
      inBatchVec = ind%batchN==batch
      X,Y,Yo = XTrain[inBatchVec],YTrain[inBatchVec],YoTrain[inBatchVec]
      val = optimizer_step(X,Y,Yo)
      tran.set_postfix(batch="%.3d"%batch, loss="%015.3f"%val.numpy())


testN = idVecTest.shape[0]
sigma2 = tf.math.softplus(sigma2_uncon)
t0SpanBefore = 12
t0SpanAfter = 48
t0Span = t0SpanBefore + t0SpanAfter
logLikelihood = logLikelihood
resArrayList = []
with tqdm(range(testN)) as progressBar:
  for indTest in progressBar:
    idTest = idVecTest[indTest]
    indIdTest = np.where(XTest[:,-1] == idTest)[0]
    XTestId,YTestId = XTest[indIdTest,:-1].copy(), YTest[indIdTest].copy()
    indNan = np.isnan(YTestId)
    YoTestId = (~indNan).astype(dtype)
    YTestId[indNan] = 0    
    tOrig = XTestId[:,timeColumnIndex] * XScale[1,timeColumnIndex] + XScale[0,timeColumnIndex]
    N = XTestId.shape[0]
    resArray = np.full([N,N+t0Span], np.nan)
    Y,Yo = tf.constant(YTestId), tf.constant(YoTestId)
    for t0Ind in range(N+t0Span):
      X = XTestId.copy()
      if t0Ind==0:
        X[:,1] = 0
        X[:,0] = 0
      else:
        X[:,1] = 1
        t0 = tOrig[0] - t0SpanBefore + t0Ind
        X[:,0] = ((tOrig-t0)-XScale[0,0])/XScale[1,0]
      X = tf.constant(X)
      logLike = logLikelihood(B,sigma2, X,Y,Yo)[1]
      resArray[:,t0Ind] = np.cumsum(logLike,0)
    resArrayList.append(resArray)

dt_early   = -12
dt_optimal = -6
dt_late    = 3
best_utilities     = np.zeros(testN)
inaction_utilities = np.zeros(testN)
with tqdm(range(testN)) as progressBar:
  for indTest in progressBar:
    idTest = idVecTest[indTest]
    indIdTest = np.where(XTest[:,-1] == idTest)[0]
    XTestId,YTestId = XTest[indIdTest,:-1].copy(), YTest[indIdTest].copy()
    sepLabelVec = ~np.isnan(XTestId[:,0]) * (XTestId[:,0]>=0.0)
    N = XTestId.shape[0]
    best_predictions     = np.zeros(N)
    inaction_predictions = np.zeros(N)
    action_predictions = np.ones(N)
    if np.any(sepLabelVec):
      t_sepsis = np.argmax(sepLabelVec) - dt_optimal
      best_predictions[max(0, t_sepsis + dt_early) : min(t_sepsis + dt_late + 1, N)] = 1
    best_utilities[indTest],_ = compute_prediction_utility(sepLabelVec, best_predictions)
    inaction_utilities[indTest],_ = compute_prediction_utility(sepLabelVec, inaction_predictions)
unnormalized_best_utility     = np.sum(best_utilities)
unnormalized_inaction_utility = np.sum(inaction_utilities)

priorProb = 0.1
dtAdvance = 6
bfThreshhold = -0.4
observed_utilities = np.zeros([testN])
with tqdm(range(testN)) as progressBar:
  for indTest in progressBar:
    idTest = idVecTest[indTest]
    resArray = resArrayList[indTest]
    indIdTest =  np.where(XTest[:,-1] == idTest)[0]
    XTestId,YTestId = XTest[indIdTest,:-1].copy(), YTest[indIdTest].copy()
    sepLabelVec = ~np.isnan(XTestId[:,0]) * (XTestId[:,0]>=0.0)
    N = XTestId.shape[0]
    isPositive = np.any(sepLabelVec)
    A = np.tri(N, N+t0Span-1, t0Span-1) - np.tri(N, N+t0Span-1, -1)
    A[A==0] = np.nan
    A *= resArray[:,1:]-resArray[:,0,None]
    B = A.copy()
    B[:,:t0SpanBefore-1] = np.nan
    neg = np.ones([N,1])
    pos = ~np.isnan(B)
    pos = pos / np.sum(pos,1)[:,None]
    C = np.concatenate([np.zeros([N,1]),B],1)
    C[np.isnan(C)] = -np.inf
    logPrior = np.log(np.concatenate([(1-priorProb)*neg,priorProb*pos],1))
    unnormPostProb = np.exp(C + logPrior)
    normPostProb = unnormPostProb / np.sum(unnormPostProb,1)[:,None]
    postProbVec = np.sum(normPostProb[:,1:]*np.tri(N,N+t0Span-1,dtAdvance+t0SpanBefore-1),1)
    postBF = logit(postProbVec)
    observed_predictions = postBF > bfThreshhold 
    observed_utilities[indTest],_ = compute_prediction_utility(sepLabelVec, observed_predictions)
       
unnormalized_observed_utility = np.sum(observed_utilities)
normalized_observed_utility = (unnormalized_observed_utility - unnormalized_inaction_utility) / (unnormalized_best_utility - unnormalized_inaction_utility)
print("\nPredictive utility score %.3f, elapsed time %ds"%(normalized_observed_utility, time() - startTime))



