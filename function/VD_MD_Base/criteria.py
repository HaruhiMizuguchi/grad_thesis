import torch
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.metrics import hamming_loss, coverage_error, label_ranking_average_precision_score, label_ranking_loss

def hLoss(output, test_target, threshold = 0.8):
    '''
    compute hamming loss
    :param output:
    :param test_target:
    :param threshold:
    :return:
    '''
    prelabel = (torch.sigmoid(output) > threshold).float()
    loss = hamming_loss(test_target.numpy(), prelabel.cpu().numpy())
    return loss

def rLoss(output, test_target):
    '''
    compute ranking loss
    :param output:
    :param test_target:
    :return:
    '''
    loss = label_ranking_loss(test_target.numpy(), output.cpu().detach().numpy())
    return loss

def oError(output, test_target):
    '''
    compute one error
    :param output:
    :param test_target:
    :return:
    '''
    y_score = output.cpu().detach().numpy()
    y_true = test_target.numpy()

    n_samples, n_labels = y_true.shape

    y_score = np.delete(y_score, np.where(np.sum(y_true, axis=1) == n_labels)[0], axis=0)
    y_true = np.delete(y_true, np.where(np.sum(y_true, axis=1) == n_labels)[0], axis=0)

    y_true = csr_matrix(y_true)
    y_score = y_score
    n_samples, n_labels = y_true.shape

    one_error = 0.0

    for i, (start, stop) in enumerate(zip(y_true.indptr, y_true.indptr[1:])):  # i 是样本的id
        relevant = y_true.indices[start:stop]
        score_i = y_score[i]
        top_one = np.where(score_i == np.max(score_i))[0]
        if len(np.intersect1d(relevant, top_one)) == 0:
            one_error += 1
    OneError = one_error / n_samples

    return OneError

def Conv(output, test_target):
    '''
    compute coverage
    :param output:
    :param test_target:
    :return:
    '''
    n_labels = test_target.shape[1] * 1.0
    loss = coverage_error(test_target.numpy(), output.cpu().detach().numpy())

    return (loss - 1) / n_labels

def avgPre(output, test_target):
    '''
    compute average precision
    :param output:
    :param test_target:
    :return:
    '''
    avgPrecision = label_ranking_average_precision_score(test_target.numpy(), output.cpu().detach().numpy())

    return avgPrecision