import torch
import numpy as np
import random
from scipy.io import loadmat
from sklearn import preprocessing
from torch.utils.data import TensorDataset, DataLoader

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
  
  #trainとtestのインデックス(自分はcv_inds)
  """indices = torch.randperm(n)
  train_idxs = indices[:train_size]
  test_idxs = indices[train_size:]"""
  index = np.where(cv_inds!=cv_num)
  index = np.ravel(index[0])
  train_idxs = torch.tensor(index)
  index = np.where(cv_inds==cv_num)
  index = np.ravel(index[0])
  test_idxs = torch.tensor(index)

  #上で指定したインデックスでtrainとtestに分ける             
  train_features = torch.index_select(features, 0, train_idxs)
  train_labels = torch.index_select(labels, 0, train_idxs)
  plabels = torch.index_select(plabels, 0, train_idxs)

  clean_features = train_features.clone()
  clean_labels = train_labels.clone()

  noisy_features = train_features.clone()
  noisy_labels = plabels.clone()

  test_features = torch.index_select(features, 0, test_idxs)
  test_labels = torch.index_select(labels, 0, test_idxs)

  #検証集合を作る
  # select a clean batch from training set
  n = len(train_features)
  # meta_size = batch_size
  indices = torch.randperm(n)

  meta_size = int(n * p_true / 100)

  meta_idxs = indices[:meta_size]
  train_idxs = indices[meta_size:]

  meta_features = torch.index_select(train_features, 0, meta_idxs)
  meta_labels = torch.index_select(train_labels, 0, meta_idxs)

  train_features = torch.index_select(train_features, 0, train_idxs)
  train_labels = torch.index_select(plabels, 0, train_idxs)

  train_dataset = TensorDataset(train_features, train_labels)
  test_dataset = TensorDataset(test_features, test_labels)
  meta_dataset = TensorDataset(meta_features, meta_labels)
  clean_dataset = TensorDataset(clean_features, clean_labels)
  noisy_dataset = TensorDataset(noisy_features, noisy_labels)

  train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
  test_loader = DataLoader(dataset=test_dataset, batch_size=test_features.size()[0], shuffle=False)
  meta_loader = DataLoader(dataset=meta_dataset, batch_size=batch_size, shuffle=True)
  clean_loader = DataLoader(dataset=clean_dataset, batch_size=batch_size, shuffle=True)
  noisy_loader = DataLoader(dataset=noisy_dataset, batch_size=batch_size, shuffle=True)

  return train_loader, test_loader, meta_loader, clean_loader, noisy_loader, features_num, labels_num