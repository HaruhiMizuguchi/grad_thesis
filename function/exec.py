from .liblinear.python.liblinear.liblinearutil import *
from .PML_NI import PML_NI
from .skmultilearn.skmultilearn.adapt import MLkNN
from .VD_MD_Base.VD_MD_Base import *

def exec_PML_NI(train_feature,train_cand,train_gt,test_feature,lambd,beta,gamma):
    clf = PML_NI.PMLNI()
    clf.fit(train_feature,train_cand,train_gt,test_feature)
    creds = clf.outputs()
    predict_labels = clf.predict()
    return creds, predict_labels

def exec_MLkNN(train_feature,train_cand,test_feature,k):
    clf = MLkNN(k=k)
    clf.fit(train_feature, train_cand)
    predict_labels = clf.predict(test_feature).toarray()
    creds = clf.predict_proba(test_feature).toarray()
    return creds, predict_labels


def exec_BR(train_feature,train_cand,test_feature,test_label):
    train_target_T = np.array(train_cand).T
    test_target_T = np.array(test_label).T
    [num_label,num_train] = np.shape(train_target_T)
    [num_test,num_fea] = np.shape(test_feature)
    #%cd /content/drive/My Drive/Colab Notebooks/function/liblinear-2.45
    creds = np.zeros((num_label,num_test))
    predict_labels = np.zeros((num_label,num_test))
    for i in range(0,num_label):
      y_train = train_target_T[i]
      y_test = test_target_T[i]
      m = train(y_train, train_feature)
      p_label, p_acc, p_val = predict(y_test, test_feature, m)
      predict_labels[i] = np.array(p_label)
      creds[i] = np.ravel(np.array(p_val))
    return creds.T, predict_labels.T

def exec_PML_VD(train_feature, train_cand, val_feature, val_gt, noise_loader, creds_loader, val_loader, test_loader, features_num, labels_num, lr):
    outputs, p_labels = run_VD_MD_Base("PML_VD", train_feature, train_cand, val_feature, val_gt, noise_loader, creds_loader, val_loader, test_loader, features_num, labels_num, lr)
    return outputs, p_labels

def exec_Baseline(train_feature, train_cand, val_feature, val_gt, noise_loader, creds_loader, val_loader, test_loader, features_num, labels_num, lr):
    outputs, p_labels = run_VD_MD_Base("Baseline", train_feature, train_cand, val_feature, val_gt, noise_loader, creds_loader, val_loader, test_loader, features_num, labels_num, lr)
    return outputs, p_labels

def exec_PML_MD(train_feature, train_cand, val_feature, val_gt, noise_loader, creds_loader, val_loader, test_loader, features_num, labels_num, lr):
    outputs, p_labels ,net = run_VD_MD_Base("PML_MD", train_feature, train_cand, val_feature, val_gt, noise_loader, creds_loader, val_loader, test_loader, features_num, labels_num, lr)
    return outputs, p_labels ,net