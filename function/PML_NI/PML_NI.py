"""入力：サンプル数 M ,特徴数 N ,ラベル数 Q とすると、

dataは M*N

train_target,true_target,test_targetは Q*M を想定

形式はnp.array"""
import numpy as np
from numpy import linalg as LA
from itertools import combinations
class PMLNI:
  def __init__(self):
    self.lambd = 10
    self.beta = 0.5
    self.gamma = 0.5
    self.max_iter = 500
  
  # train_data:M*D (num_data * num_feature)
  # train_target:M*Q (num_label * num_data)
  # true_target:M*Q (num_label * num_data)
  # test_data:N*D (num_data * num_feature)
  def fit(self,train_data,train_target,true_target,test_data):
    lambd = self.lambd
    beta = self.beta
    gamma = self.gamma
    max_iter = self.max_iter

    train_target_T = train_target
    true_target_T = true_target
    train_target = train_target_T.T
    true_target = true_target_T.T
    [num_train,dim] = np.shape(train_data)
    [num_label,a] = np.shape(train_target)
    ##Training
    fea_matrix = np.concatenate([train_data,np.ones((num_train,1))],1)
    U = np.zeros((num_label,dim+1))
    V = np.zeros((num_label,dim+1))
    Y = np.zeros((num_label,dim+1))
    mu = 1e-4
    rho = 1.1

    YX = train_target @ fea_matrix
    XX = fea_matrix.T @ fea_matrix

    for t in range(0,max_iter):
      # update W
      
      # W = (YX+(mu*U)+(mu*V)+Y)/(XX+(lambd+mu)*np.eye(dim+1))
      A = YX+(mu*U)+(mu*V)+Y
      B = XX+(lambd+mu)*np.eye(dim+1)
      W = np.linalg.lstsq(B.T,A.T)[0].T
      #W = np.matmul(A,np.linalg.inv(B))

      # update U V
      Uk = np.copy(U)
      Vk = np.copy(V)
      [M,s,Nhat] = np.linalg.svd(W-V-Y/mu, full_matrices=False)
      svp = 0
      for i in range(0,len(s)):
        if s[i] > beta/mu:
          svp += 1
      if svp >= 1:
        s = s[0:svp] - beta/mu
      else:
        svp = 1
        s = np.array([0])
      Uhat =M[:,0:svp] @ np.diag(s) @ Nhat[0:svp,:]
      U = np.copy(Uhat)
      
      #L1 NORM
      Vraw = W - U - Y/mu
      Vhat = np.zeros((num_label,dim+1))
      for i in range(0,num_label):
        for j in range(0,dim+1):
          Vhat[i,j] = max(Vraw[i,j]-gamma/mu,0) + min(Vraw[i,j]+gamma/mu,0)
      
      '''# L2,1 NORM
      Vraw = W - U - Y/mu
      Vhat = np.zeros((num_label,dim+1))
      for j in range(0,num_label):
        v = Vraw[j,:]
        vNorm = norm(v)
        if vNorm > gamma/mu:
          Vhat[j,:] = (vNorm-gamma/mu) / vNorm * v'''
      V = np.copy(Vhat)

      # stop criterion
      convg2 = False
      stopCriterion2 = mu*LA.norm(U-Uk,'fro')/LA.norm(W,'fro')
      if stopCriterion2 < 1e-5:
        convg2=True
      convg1 = False
      tmp = W - U - V
      stopCriterion1 = LA.norm(tmp,'fro')/LA.norm(W,'fro')
      if stopCriterion1 < 1e-7:
        convg1 = True
      if convg2:
        mu = min(rho*mu,1e10)
      Y = Y + mu*(U+V-W)

      if convg1 and convg2:
        break
    
    # Computing the size predictor using linear least squares model
    # しきい値の予測器の計算
    Outputs = fea_matrix @ U.T # M*Q
    Left = Outputs
    Right = np.zeros((num_train,1)) # M*1
    for i in range(0,num_train):
      temp = Left[i,:]
      index = np.argsort(temp)
      temp_sorted = np.sort(temp)
      candidate = np.zeros((1,num_label + 1))
      candidate[0,0] = temp[0]-0.1
      for j in range(0,num_label-1):
        candidate[0,j+1] = (temp_sorted[j]+temp_sorted[j+1])/2
      candidate[0,num_label] = temp_sorted[num_label - 1] + 0.1
      miss_class = np.zeros((1,num_label+1))
      for j in range(0,num_label+1):
        temp_notlabels = index[:j]
        temp_labels = index[j:]
        notlabels_true_target = np.ravel(np.where(true_target_T[i,:] != 1))
        labels_true_target = np.ravel(np.where(true_target_T[i,:] == 1))
        false_neg = len(np.array(list(set(temp_notlabels)-set(notlabels_true_target))))
        false_pos = len(np.array(list(set(temp_labels)-set(labels_true_target))))
        miss_class[0,j] = false_neg + false_pos
      temp_index = np.argmin(miss_class)
      Right[i,0] = candidate[0,temp_index]
    Left = np.concatenate([Left,np.ones((num_train,1))],1)

    # for matlab : tempvalue = (Left\Right).T
    tempvalue = np.linalg.lstsq(Left, Right)[0]
    '''num_vars = Left.shape[1]
    rank = np.linalg.matrix_rank(Left)
    if rank == num_vars:
        tempvalue = np.linalg.lstsq(Left, Right)[0]    # not under-determined
    else:
        for nz in combinations(range(num_vars), rank):    # the variables not set to zero
            try: 
                tempvalue = np.zeros((num_vars, 1))  
                tempvalue[nz, :] = np.asarray(np.linalg.solve(Left[:, nz], Right))
            except np.linalg.LinAlgError:     
                pass'''
    Weights_sizepre = tempvalue[0:num_label]
    Bias_sizepre = tempvalue[num_label:]

    #compute Treshold for each test data
    [num_test, a] = np.shape(test_data)
    Outputs = np.concatenate([test_data,np.ones((num_test,1))],1) @ U.T
    WX = test_data @ U[:,:-1].T
    Threshold = np.concatenate([WX,np.ones((num_test,1))],1) @ np.concatenate([Weights_sizepre,Bias_sizepre])

    # compute result
    Pre_Labels = np.zeros((num_test,num_label))
    for i in range(0,num_test):
      for k in range(0,num_label):
        if Outputs[i,k] >= Threshold[i,0]:
          Pre_Labels[i,k] = 1
        else:
          Pre_Labels[i,k] = 0
    
    self.Outputs = Outputs
    self.Labels = Pre_Labels

  def outputs(self):
    return(self.Outputs)

  def predict(self):
    return(self.Labels)