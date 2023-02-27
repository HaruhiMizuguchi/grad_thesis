### インスタンス数 N, 特徴数 D とすると特徴量の入力は N×D の行列
### クラス数 Q とするとラベルの入力は N×Q の行列(ラベルを持つとき1, 持たない時は0)

from function.skmultilearn import skmultilearn
import sys
sys.path.append("function/skmultilearn")
sys.path.append("function/VD_MD_Base")
from function.exec import *
from function.utils.process_data import *
from function.utils import process_data
from function.metrics.metrics_new import *


### ここを変更して実行箇所を変える
### 実行する手法
method_list = ["PML-VD","Baseline","PML-MD","PML-NI","BR","ML-kNN"]
### 実行するデータ
data_list = ['music_style','mirflickr','emotions','enron','emotions','CAL500','scene','genbase']
### マルチラベルデータセットの擬陽性ラベルの量(%)
noise_list_multi = [50,100,150]
### 検証集合サイズの訓練データに対する割合(%)
val_list = [1,3,5,7,9]
### 5分割交差検証を実行する箇所
cv_start = 0
cv_end = 5

method_list = ["PML-NI"]
data_list = ['emotions']
noise_list_multi = [50]
val_list = [1]
cv_start = 0
cv_end = 1


for method in method_list:
    for data in data_list:
        if data in ['music_style','mirflickr','music_emotion']:
            noise_list = [0]
        else:
            noise_list = noise_list_multi
        for p_noise in noise_list:
            for p_val in val_list:
                for cv in range(cv_start,cv_end):
                    print(f"start method={method}, data={data}, noise={str(p_noise)}, val={str(p_val)}, cv={str(cv)}")
                    ### データの読み込み
                    train_feature,train_cand,train_gt,test_feature,test_gt,train_val_feature,train_val_gt,train_noise_feature,train_noise_cand,batch_size = \
                        read_data(data,p_noise,p_val,cv)

                    ### ニューラルネット用のデータ
                    if method in ["PML-VD","Baseline","PML-MD"]:
                        if method == "PML-VD":
                            creds_pre = predict_creds(train_val_feature,train_val_gt,train_noise_feature,train_noise_cand)
                        else: 
                            creds_pre = train_noise_cand
                        
                        noise_loader, creds_loader, val_loader, test_loader, features_num, labels_num = \
                            get_dataloader(train_feature,train_cand,train_gt,test_feature,test_gt,train_val_feature,train_val_gt,train_noise_feature,train_noise_cand,creds_pre,batch_size)


                    ### 実行
                    if method == "PML-VD":

                        ### パラメータ
                        lr = 0.03 #学習率

                        creds, predict_labels = exec_PML_VD(train_noise_feature, train_noise_cand, train_val_feature, train_val_gt, noise_loader, creds_loader, val_loader, test_loader, features_num, labels_num, lr)

                    elif method == "Baseline":
                        
                        ### パラメータ
                        lr = 0.03 #学習率

                        creds, predict_labels = exec_Baseline(train_noise_feature, train_noise_cand, train_val_feature, train_val_gt, noise_loader, creds_loader, val_loader, test_loader, features_num, labels_num, lr)
                    
                    elif method == "PML-MD":

                        ### パラメータ
                        lr = 0.03 #学習率

                        creds, predict_labels, net = exec_PML_MD(train_noise_feature, train_noise_cand, train_val_feature, train_val_gt, noise_loader, creds_loader, val_loader, test_loader, features_num, labels_num, lr)
                    
                    elif method == "PML-NI":

                        ### パラメータ
                        lambd = 10
                        beta = 0.5
                        gamma = 0.5

                        creds, predict_labels = exec_PML_NI(train_feature,train_cand,train_gt,test_feature,lambd,beta,gamma)
                    
                    elif method == "BR":

                        ### パラメータなし
                        creds, predict_labels = exec_BR(train_feature,train_cand,test_feature,test_gt)
                    
                    elif method == "ML-kNN":

                        ### パラメータ
                        k=3

                        creds, predict_labels = exec_MLkNN(train_feature,train_cand.astype(int),test_feature,k)
                    
                    else:
                        print("this method is not supported.")

                    ### 評価の表示
                    print(f"finish method={method}, data={data}, noise={str(p_noise)}, val={str(p_val)}, cv={str(cv)}")
                    eval_and_print(creds,predict_labels,test_gt)