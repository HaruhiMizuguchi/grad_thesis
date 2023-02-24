import torch
import time
import os
import torch.optim as optim
import numpy as np
from model import LinearNet
from dataset import get_loader
from scipy.io import savemat
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
    net.eval()

    hLoss_list = []
    rLoss_list = []
    oError_list = []
    conv_list = []
    avgPre_list = []

    for itr, (inputs, labels) in enumerate(test_loader):

        inputs = inputs.to(device)
        outputs = net(inputs)

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

    return hamming_loss, ranking_loss, one_error, coverage, avg_precision

def train_common(net, optimizer, train_loader):
    train_loss = 0

    for batch_idx, (inputs, labels) in enumerate(train_loader):
        net.train()
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = net(inputs)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    return train_loss

def train_meta(net, optimizer, train_loader, meta_loader):
    train_loss = 0.0
    meta_loss = 0.0
    clean_iter = iter(meta_loader)
    for batch_idx, (inputs, labels) in enumerate(train_loader):
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

        try:
            val_data, val_labels = next(clean_iter)
        except StopIteration:
            clean_iter = iter(meta_loader)
            val_data, val_labels = next(clean_iter)

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

        y_f_hat = net(inputs)
        l_f = criterion(y_f_hat, p)

        optimizer.zero_grad()
        l_f.backward()
        optimizer.step()

        meta_loss += l_g_meta.item()
        train_loss += l_f.item()

    return meta_loss, train_loss


def baseline(train_loader, test_loader, meta_loader, num_epoch=500, clean=True):
    # method = baseline for baseline with meta data, else for baseline without meta data and ground-truth

    net = LinearNet(num_inputs=features_num, num_outputs=labels_num)
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

        hamming_loss, ranking_loss, one_error, coverage, avg_precision = test(net, test_loader)

        print('Batch: [{:0>4}/{:0>4}] '.format(epoch + 1, num_epoch),
              'training loss: {:.4f} '.format(loss),
              'hLoss: {:.4f} '.format(hamming_loss),
              'rLoss: {:.4f} '.format(ranking_loss),
              'oError: {:.4f} '.format(one_error),
              'conv: {:.4f} '.format(coverage),
              'avgPre: {:.4f}'.format(avg_precision))

        hLoss_list.append(hamming_loss)
        rLoss_list.append(ranking_loss)
        oError_list.append(one_error)
        conv_list.append(coverage)
        avgPre_list.append(avg_precision)
        
    return np.min(hLoss_list), np.min(rLoss_list), np.min(oError_list), np.min(conv_list), np.max(avgPre_list)


def metalearning(train_loader, test_loader, meta_loader, num_epoch=500, clean=True):

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
            train_l, meta_l = train_meta(net, optimizer, train_loader, meta_loader)
            train_loss += train_l
            meta_loss += meta_l

            train_loss /= (len(train_loader) + len(meta_loader))
            meta_loss /= len(train_loader)
        else:
            train_l, meta_l = train_meta(net, optimizer, train_loader, meta_loader)
            train_loss += train_l
            meta_loss += meta_l

            train_loss /= len(train_loader)
            meta_loss /= len(train_loader)

        hamming_loss, ranking_loss, one_error, coverage, avg_precision = test(net, test_loader)

        print('Batch: [{:0>4}/{:0>4}] '.format(epoch + 1, num_epoch),
              'training loss: {:.4f} '.format(train_loss),
              'meta_loss: {:.4f} '.format(meta_loss),
              'hLoss: {:.4f} '.format(hamming_loss),
              'rLoss: {:.4f} '.format(ranking_loss),
              'oError: {:.4f} '.format(one_error),
              'conv: {:.4f} '.format(coverage),
              'avgPre: {:.4f}'.format(avg_precision))

        hLoss_list.append(hamming_loss)
        rLoss_list.append(ranking_loss)
        oError_list.append(one_error)
        conv_list.append(coverage)
        avgPre_list.append(avg_precision)

    return np.min(hLoss_list), np.min(rLoss_list), np.min(oError_list), np.min(conv_list), np.max(avgPre_list)


def save_result(data, method, learning_rate, result):
    file = './result/LinearNet/' + data
    if not os.path.exists(file):
        os.mkdir(file)
    filename = './result/LinearNet/' + data + '/' + method + '_' + str(learning_rate) + '.mat'
    content = {}

    content['result'] = result
    savemat(filename, content)

def save_log(content, learning_rate):
    filename = './result/LinearNet/' + str(learning_rate) + '_log.txt'
    file = open(filename, 'a')
    file.write(content)
    file.close()

def adjust_learning_rate(optimizer, epochs, learning_rate):
    lr = learning_rate * ((0.1 ** int(epochs >= 80)) * (0.1 ** int(epochs >= 100)))  # For WRN-28-10
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

batch_size = 100
meta_size = 50
#lr_list = [0.01, 0.001]
lr_list = [0.01]
momentum = 0.9
weight_decay = 1e-4
methods = ['meta', 'meat_clean', 'baseline', 'baseline_clean', 'ground_truth']
datasets = ['cal500', 'corel5k', 'emotions', 'enron', 'image', 'mediamill', 'medical', 'scene', 'slashdot', 'tmc', 'yeast']
repeat = 1
noise_list = [50]
true_list = [1]

if __name__ == '__main__':

    #for data in datasets:
    for data in ["emotions"]:
        print('\n')

        if data in ['music_emotion', 'music_style', 'corel5k', 'slashdot']:
            batch_size = 200
        elif data in ['mirflickr', 'tmc', 'mediamill']:
            batch_size = 500
        elif data in ['cal500', 'emotions']:
            batch_size = 50

        for lr in lr_list:
            for method in methods:
                print('Dataname: {}\t Methods: {}\t Learning rate: {}'.format(data, method, lr))
                result = np.empty((repeat, 5), dtype=np.float)
                for i in range(repeat):
                    for p_noise in noise_list:
                        for p_true in true_list:
                            cv_num = i
                            print('\nRepeat {}: '.format(i + 1))
                            train_loader, test_loader, meta_loader, clean_loader, \
                            noisy_loader, features_num, labels_num = get_loader(data, batch_size, cv_num,p_noise,p_true,meta_size=meta_size)

                            if method == 'meta':
                                best_hLoss, best_rLoss, best_oError, best_conv, best_avgPre = metalearning(train_loader,
                                                                                                        test_loader,
                                                                                                        meta_loader,
                                                                                                        clean=False)
                            elif method == 'meta_clean':
                                best_hLoss, best_rLoss, best_oError, best_conv, best_avgPre = metalearning(train_loader,
                                                                                                        test_loader,
                                                                                                        meta_loader,
                                                                                                        clean=True)
                            elif method == 'baseline':
                                best_hLoss, best_rLoss, best_oError, best_conv, best_avgPre = baseline(noisy_loader,
                                                                                                    test_loader, meta_loader,
                                                                                                    clean=False)
                            elif method == 'baseline_clean':
                                best_hLoss, best_rLoss, best_oError, best_conv, best_avgPre = baseline(train_loader,
                                                                                                    test_loader, meta_loader,
                                                                                                    clean=True)

                            elif method == 'ground_truth':           # ground truth
                                best_hLoss, best_rLoss, best_oError, best_conv, best_avgPre = baseline(clean_loader,
                                                                                                    test_loader, meta_loader,
                                                                                                    clean=False)

                            result[i, :] = best_hLoss, best_rLoss, best_oError, best_conv, best_avgPre

                            print()
                            print('Test results of the best found model :\t',
                                'Best hLoss: {:.4f} '.format(best_hLoss),
                                'Best rLoss: {:.4f} '.format(best_rLoss),
                                'Best oError: {:.4f} '.format(best_oError),
                                'Best conv: {:.4f} '.format(best_conv),
                                'Best avgPre: {:.4f} '.format(best_avgPre))

                result_mean = np.mean(result, axis=0)
                result_std = np.std(result, axis=0)

                content = time.strftime('%Y-%m-%d %H:%M:%S   ', time.localtime()) + \
                                        'Dataset: {:15}  '.format(data) + 'Method: {:15}  '.format(method) + \
                                        'Learning Rate: {:.4f} '.format(lr) +\
                                        'hLoss: {:.4f}/{:.4f}  '.format(result_mean[0], result_std[0]) + \
                                        'rLoss: {:.4f}/{:.4f}  '.format(result_mean[1], result_std[1]) + \
                                        'oError: {:.4f}/{:.4f}  '.format(result_mean[2], result_std[2]) + \
                                        'conv: {:.4f}/{:.4f}  '.format(result_mean[3], result_std[3]) + \
                                        'avgPre: {:.4f}/{:.4f}\n'.format(result_mean[4], result_std[4])
                print()
                print(content)

                save_result(data, method, lr, result)
                save_log(content, lr)
