import torch
import numpy as np
import random
from scipy.io import loadmat
from sklearn import preprocessing
from torch.utils.data import TensorDataset, DataLoader
from numpy import linalg as LA
import csv
from scipy.optimize import nnls
from sklearn.preprocessing import MinMaxScaler

def regu(a):
  a = a.T
  scaler = MinMaxScaler(feature_range=(0, 1), copy=True)
  scaler.fit(a.astype('float'))
  aa = scaler.transform(a.astype('float')).T
  return(aa)

def predict_creds(meta_features,meta_labels,train_features,train_labels):
  meta_features_n=meta_features.to('cpu').detach().numpy().copy()
  meta_labels_n=meta_labels.to('cpu').detach().numpy().copy()
  train_features_n=train_features.to('cpu').detach().numpy().copy()
  train_labels_n=train_labels.to('cpu').detach().numpy().copy()
  W = np.zeros((train_features_n.shape[0],meta_features_n.shape[0]))
  for i in range(train_features_n.shape[0]):
      W[i] = nnls(meta_features_n.T,train_features_n[i])[0]
  print(f"W:{str(np.shape(W))}\ntrain:{str(np.shape(train_features_n))}\nmeta:{str(np.shape(meta_features_n))}")
  # 重みを用いて訓練集合の信頼度を推定
  creds_pre = (meta_labels_n.T @ W.T).T
  creds_pre[np.where(train_labels_n==0)] = 0
  creds_pre = regu(creds_pre)
  np.savetxt("/content/drive/MyDrive/creds_musicemotion.csv",creds_pre,delimiter=',')
  return torch.tensor(creds_pre, dtype=torch.float)
  

def feature_normalize(data):
    '''
    Normalize features by MinMaxScaler.
    :param data: unnormalized features 
    :return data: normalized features
    '''
    scaler = preprocessing.MinMaxScaler(feature_range=(-1.0, 1.0)).fit(data)
    data = scaler.transform(data)

    return data

def generate_noisy_labels(labels):

    N, C = labels.shape

    alpha = []
    for i in range(C):
        alpha.append(round(random.uniform(0.5, 0.9), 1))

    alpha = np.array(alpha)
    alpha_mat = np.tile(alpha, (N, 1))
    rand_mat = np.random.rand(N, C)
    

    mask = np.zeros((N, C), dtype=np.float)
    mask[labels!=1] = rand_mat[labels!=1] < alpha_mat[labels!=1]
    noisy_labels = labels.clone()
    noisy_labels[mask==1] = -noisy_labels[mask==1]

    return noisy_labels.numpy(), labels.numpy()

# def get_loader(dataname, batch_size, meta_size=50, prec=0.8):
def get_loader(dataname,batch_size,cv_num,p_noise,p_true,meta_size=50):
  #環境によって変える。kind_of_dataのディレクトリ
  data_path = f"/content/drive/MyDrive/Colab Notebooks/new_data2/" + dataname + "/"
  features = np.loadtxt(data_path+"data.csv", delimiter=',')
  labels = np.loadtxt(data_path+"target.csv", delimiter=',',dtype = float)
  if dataname in ['mirflickr', 'music_emotion', 'music_style','YeastBP']:
    plabels = np.loadtxt(data_path+"cand/0.csv", delimiter=',',dtype = float)
  else:
    plabels = np.loadtxt(data_path+"cand/"+str(p_noise)+".csv", delimiter=',',dtype = float)
  cv_inds = np.loadtxt(data_path+"index/5-cv.csv",delimiter=',',dtype = int)-1
  data_path_true = "/content/drive/MyDrive/Colab Notebooks/new_data/" + dataname + "/index/true/" + str(p_true) +  "/A.csv"
  with open(data_path_true) as f:
      reader = csv.reader(f)
      l = [row for row in reader]
  meta_inds = [[int(v) for v in row] for row in l]
  
  features_num = features.shape[1]
  labels_num = labels.shape[1]

  #全体の特徴をtensorに変換
  features = torch.tensor(features, dtype=torch.float)
  labels = torch.tensor(labels, dtype=torch.float)
  plabels = torch.tensor(plabels, dtype=torch.float)

  zeros = torch.zeros_like(labels)
  labels = torch.where(labels == -1, zeros, labels)
  plabels = torch.where(plabels == -1, zeros, plabels)

  #訓練とテストに分割(自分は5分割交差検証)
  # split training into train and test set
  """n = len(features)
  train_size = int(n * prec)"""
  
  #trainとtestとmetaのインデックス(自分はcv_inds)
  """indices = torch.randperm(n)
  train_idxs = indices[:train_size]
  test_idxs = indices[train_size:]"""
  index = np.where(cv_inds!=cv_num)
  train_inds = np.ravel(index[0])
  train_idxs = torch.tensor(train_inds)
  
  index = np.where(cv_inds==cv_num)
  index = np.ravel(index[0])
  test_idxs = torch.tensor(index)
  

  #上で指定したインデックスでtrainとtestに分ける             
  train_features = torch.index_select(features, 0, train_idxs)
  train_labels = torch.index_select(labels, 0, train_idxs)
  train_plabels = torch.index_select(plabels, 0, train_idxs)

  clean_features = train_features.clone()
  clean_labels = train_labels.clone()

  noisy_features = train_features.clone()
  noisy_labels = train_plabels.clone()

  test_features = torch.index_select(features, 0, test_idxs)
  test_labels = torch.index_select(labels, 0, test_idxs)

  
  #検証集合を作る
  # select a clean batch from training set
  meta_idxs = np.array([],dtype = "int")
  for i in range(5):
    if i != cv_num:
      meta_idxs = np.append(meta_idxs,np.array(meta_inds[i],dtype="int"))

  if dataname in ["music_emotion","music_style"]:
    batch_size = 200
  elif dataname in ["mirflickr"]:
    batch_size = 500
  elif dataname in ['CAL500','emotions','genbase']:
    batch_size = 50
  elif dataname in ['scene','enron']:
    batch_size = 100

  train_idxs = np.setdiff1d(train_inds,meta_idxs)
  train_idxs = torch.tensor(train_idxs)
  meta_idxs = torch.tensor(meta_idxs)

  meta_features = torch.index_select(features, 0, meta_idxs)
  meta_labels = torch.index_select(labels, 0, meta_idxs)

  
  train_features = torch.index_select(features, 0, train_idxs)
  train_labels = torch.index_select(plabels, 0, train_idxs)

  train_creds = predict_creds(meta_features,meta_labels,train_features,train_labels)
  
  train_dataset = TensorDataset(train_features, train_labels,train_labels)
  train_dataset_with_creds = TensorDataset(train_features,train_labels,train_creds)
  test_dataset = TensorDataset(test_features, test_labels)
  meta_dataset = TensorDataset(meta_features, meta_labels, meta_labels)
  clean_dataset = TensorDataset(clean_features, clean_labels, clean_labels)
  noisy_dataset = TensorDataset(noisy_features, noisy_labels, noisy_labels)

  train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
  creds_loader = DataLoader(dataset=train_dataset_with_creds,batch_size=batch_size, shuffle=True)
  test_loader = DataLoader(dataset=test_dataset, batch_size=test_features.size()[0], shuffle=False)
  meta_loader = DataLoader(dataset=meta_dataset, batch_size=batch_size, shuffle=True)
  clean_loader = DataLoader(dataset=clean_dataset, batch_size=batch_size, shuffle=True)
  noisy_loader = DataLoader(dataset=noisy_dataset, batch_size=batch_size, shuffle=True)

  return train_loader, creds_loader, test_loader, meta_loader, clean_loader, noisy_loader, features_num, labels_num