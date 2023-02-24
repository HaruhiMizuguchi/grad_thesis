import torch
import time
import os
import torch.optim as optim
import numpy as np
import csv
from model import LinearNet,Network
from criteria import hLoss, rLoss, oError, Conv, avgPre

device = ('cuda' if torch.cuda.is_available() else 'cpu')

def criterion(predictions, confidence):

    assert predictions.shape == confidence.shape

    N, C = confidence.shape

    loss_sum = torch.zeros(N, 1, dtype=torch.float, device=device)

    ones = torch.ones_like(confidence, dtype=torch.float, device=device)
    zeros = torch.zeros_like(confidence, dtype=torch.float, device=device)

    for i in range(C):
        confidence_i = confidence[:, i].view(N, -1)
        prediction_i = predictions[:, i].view(N, -1)

        loss = torch.max(zeros, confidence_i - confidence) * torch.max(zeros, ones - (prediction_i - predictions))
        loss_sum += torch.sum(loss, dim=-1, keepdim=True) / ((C - 1) * 1.0)

    return loss_sum.mean()

def test(net, test_loader):

    ######################################
    #信頼度のしきい値
    threshold = 0.8
    ######################################

    net.eval()

    hLoss_list = []
    rLoss_list = []
    oError_list = []
    conv_list = []
    avgPre_list = []

    for itr, (inputs, labels) in enumerate(test_loader):

        inputs = inputs.to(device)
        outputs = net(inputs)
        prelabel = (torch.sigmoid(outputs) > threshold).float()

        hLoss_list.append(hLoss(outputs, labels))
        rLoss_list.append(rLoss(outputs, labels))
        oError_list.append(oError(outputs, labels))
        conv_list.append(Conv(outputs, labels))
        avgPre_list.append(avgPre(outputs, labels))

    hamming_loss = np.mean(hLoss_list)
    ranking_loss = np.mean(rLoss_list)
    one_error = np.mean(oError_list)
    coverage = np.mean(conv_list)
    avg_precision = np.mean(avgPre_list)

    return hamming_loss, ranking_loss, one_error, coverage, avg_precision, outputs.cpu().detach().numpy() ,prelabel.cpu().numpy()

def train_common(net, optimizer, train_loader):
    train_loss = 0

    for batch_idx, (inputs, labels, creds) in enumerate(train_loader):
        net.train()
        inputs, labels, creds = inputs.to(device), labels.to(device), creds.to(device)
        outputs = net(inputs)
        loss = criterion(outputs, creds)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    return train_loss

def train_meta(net, optimizer, train_loader, meta_loader, features_num, labels_num, lr):
    train_loss = 0.0
    meta_loss = 0.0
    clean_iter = iter(meta_loader)
    for batch_idx, (inputs, labels,creds) in enumerate(train_loader):
        inputs, labels,creds = inputs.to(device), labels.to(device),creds.to(device)
        meta_net = LinearNet(num_inputs=features_num, num_outputs=labels_num)
        meta_net.load_state_dict(net.state_dict())
        meta_net.to(device)
        epsilon = torch.zeros_like(labels, requires_grad=True)

        y_f_hat = meta_net(inputs)
        l_f_meta = criterion(y_f_hat, epsilon)
        meta_net.zero_grad()

        grads = torch.autograd.grad(l_f_meta, (meta_net.params()), create_graph=True)
        meta_net.update_params(lr, source_params=grads)

        try:
            val_data, val_labels,val_creds = next(clean_iter)
        except StopIteration:
            clean_iter = iter(meta_loader)
            val_data, val_labels,val_creds = next(clean_iter)

        val_data, val_labels = val_data.to(device), val_labels.to(device)

        y_g_hat = meta_net(val_data)
        l_g_meta = criterion(y_g_hat, val_labels)

        grad_eps = torch.autograd.grad(l_g_meta, epsilon, only_inputs=True)[0]

        # computing and normalizing the confidence matrix P
        # p_tilde = torch.clamp(epsilon - grad_eps, min=0)
        p_tilde = torch.clamp(-grad_eps, min=0)
        p_tilde *= labels

        max_row = torch.max(p_tilde, dim=1, keepdim=True)[0]
        ones = torch.ones_like(max_row)
        max_row = torch.where(max_row == 0, ones, max_row)
        p = p_tilde / max_row
        # p.shape = batch_size * class_num

        y_f_hat = net(inputs)
        l_f = criterion(y_f_hat, p)

        optimizer.zero_grad()
        l_f.backward()
        optimizer.step()

        meta_loss += l_g_meta.item()
        train_loss += l_f.item()

    return meta_loss, train_loss


def baseline(train_loader, test_loader, meta_loader, features_num, labels_num, lr, momentum = 0.9, weight_decay = 1e-4, num_epoch=300,clean=True):
    # method = baseline for baseline with meta data, else for baseline without meta data and ground-truth

    net = LinearNet(num_inputs=features_num, num_outputs=labels_num)
    print(features_num)
    print(labels_num)
    #net = Network(num_inputs=features_num,num_hides = 5,num_outputs=labels_num)
    net = net.to(device)
    optimizer = optim.SGD(params=net.params(), lr=lr, momentum=momentum, weight_decay=weight_decay)

    hLoss_list = []
    rLoss_list = []
    oError_list = []
    conv_list = []
    avgPre_list = []

    for epoch in range(num_epoch):
        loss = 0.0
        # adjust_learning_rate(optimizer, epoch, lr)
        if clean:
            loss += train_common(net, optimizer, meta_loader)
            loss += train_common(net, optimizer, train_loader)
            loss /= (len(meta_loader) + len(train_loader))
        else:
            loss += train_common(net, optimizer, train_loader)
            loss /= len(train_loader)

        hamming_loss, ranking_loss, one_error, coverage, avg_precision, outputs, p_labels = test(net, test_loader)

        print('Batch: [{:0>4}/{:0>4}] '.format(epoch + 1, num_epoch),
              'training loss: {:.4f} '.format(loss),
              'hLoss: {:.4f} '.format(hamming_loss),
              'rLoss: {:.4f} '.format(ranking_loss),
              'oError: {:.4f} '.format(one_error),
              'conv: {:.4f} '.format(coverage),
              'avgPre: {:.4f}'.format(avg_precision))

    #最終的な結果と出力と予測ラベルを返す
    return hamming_loss, ranking_loss, one_error, coverage, avg_precision, outputs, p_labels

def metalearning(train_loader, test_loader, meta_loader,train_data,train_plabels, meta_features, meta_labels,features_num,labels_num, lr, momentum=0.9, weight_decay=1e-4, num_epoch=300, clean=True,save_creds_every_epoch=False):
    
    net = LinearNet(num_inputs=features_num, num_outputs=labels_num)
    net = net.to(device)
    optimizer = optim.SGD(params=net.params(), lr=lr, momentum=momentum, weight_decay=weight_decay)

    hLoss_list = []
    rLoss_list = []
    oError_list = []
    conv_list = []
    avgPre_list = []

    for epoch in range(num_epoch):
        train_loss = 0.0
        meta_loss = 0.0

        if clean:
            train_loss += train_common(net, optimizer, train_loader)
            train_l, meta_l = train_meta(net, optimizer, train_loader, meta_loader, features_num, labels_num, lr)
            train_loss += train_l
            meta_loss += meta_l

            train_loss /= (len(train_loader) + len(meta_loader))
            meta_loss /= len(train_loader)
        else:
            train_l, meta_l = train_meta(net, optimizer, train_loader, meta_loader, features_num, labels_num, lr)
            train_loss += train_l
            meta_loss += meta_l

            train_loss /= len(train_loader)
            meta_loss /= len(train_loader)

        """if save_creds_every_epoch == True:
          p_creds = compute_p_creds(train_data,train_plabels, meta_features, meta_labels,net)
          np.savetxt(f"/content/drive/MyDrive/Colab Notebooks/result/predict_creds/PMLMD_every_epoch/{data}/{str(p_true)}/{epoch}.csv",p_creds,delimiter = ",")
        """

        hamming_loss, ranking_loss, one_error, coverage, avg_precision ,outputs, p_labels= test(net, test_loader)

        print('Batch: [{:0>4}/{:0>4}] '.format(epoch + 1, num_epoch),
              'training loss: {:.4f} '.format(train_loss),
              'meta_loss: {:.4f} '.format(meta_loss),
              'hLoss: {:.4f} '.format(hamming_loss),
              'rLoss: {:.4f} '.format(ranking_loss),
              'oError: {:.4f} '.format(one_error),
              'conv: {:.4f} '.format(coverage),
              'avgPre: {:.4f}'.format(avg_precision))

    #最終的な結果と出力と予測ラベルを返す
    return hamming_loss, ranking_loss, one_error, coverage, avg_precision, outputs, p_labels, net

#メタ学習時の信頼度の推定
def compute_p_creds(train_data,train_plabels, meta_features, meta_labels, net):
    # 最終的なネットワークをもとに訓練データ全体の信頼度を計算
    inputs = train_data
    labels = train_plabels
    inputs, labels = inputs.to(device), labels.to(device)

    meta_net = LinearNet(num_inputs=features_num, num_outputs=labels_num)
    meta_net.load_state_dict(net.state_dict())
    meta_net.to(device)
    epsilon = torch.zeros_like(labels, requires_grad=True)

    y_f_hat = meta_net(inputs)
    l_f_meta = criterion(y_f_hat, epsilon)
    meta_net.zero_grad()

    grads = torch.autograd.grad(l_f_meta, (meta_net.params()), create_graph=True)
    meta_net.update_params(lr, source_params=grads)

    val_data = meta_features
    val_labels = meta_labels
    val_data, val_labels = val_data.to(device), val_labels.to(device)

    y_g_hat = meta_net(val_data)
    l_g_meta = criterion(y_g_hat, val_labels)

    grad_eps = torch.autograd.grad(l_g_meta, epsilon, only_inputs=True)[0]

    # computing and normalizing the confidence matrix P
    # p_tilde = torch.clamp(epsilon - grad_eps, min=0)
    p_tilde = torch.clamp(-grad_eps, min=0)
    p_tilde *= labels

    max_row = torch.max(p_tilde, dim=1, keepdim=True)[0]
    ones = torch.ones_like(max_row)
    max_row = torch.where(max_row == 0, ones, max_row)
    p = p_tilde / max_row
    p = p.cpu().numpy()
    return p

def save_p_creds(data,train_target,p_noise,p_true,cv_num,creds):
  save_path = f"/content/drive/MyDrive/Colab Notebooks/result/predict_creds/PML-MD/{data}/{str(p_noise)}/true/{str(p_true)}/"
  np.savetxt(save_path+"creds_"+str(cv_num)+".csv",creds,delimiter=",")
  np.savetxt(save_path+"target_"+str(cv_num)+".csv",train_target,delimiter=",")

def run_VD_MD_Base(method, train_feature, train_cand, val_feature, val_gt, noise_loader, creds_loader, val_loader, test_loader, features_num, labels_num, lr):

    if method == "PML_VD":
        hamming_loss, ranking_loss, one_error, coverage, avg_precision ,outputs, p_labels = \
            baseline(creds_loader, test_loader, val_loader, features_num, labels_num, lr, clean=True)
        return outputs, p_labels

    elif method == "Baseline":
        hamming_loss, ranking_loss, one_error, coverage, avg_precision ,outputs, p_labels = \
            baseline(noise_loader, test_loader, val_loader, features_num, labels_num, lr, clean=False)
        return outputs, p_labels

    elif method == "PML_MD":
        hamming_loss, ranking_loss, one_error, coverage, avg_precision ,outputs, p_labels ,net = \
            metalearning(noise_loader, test_loader, val_loader, train_feature, train_cand, val_feature, val_gt, features_num, labels_num, lr, clean=False, save_creds_every_epoch=False)
        return outputs, p_labels, net
        