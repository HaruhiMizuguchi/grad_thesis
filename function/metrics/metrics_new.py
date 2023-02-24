import numpy as np
from scipy.sparse import csr_matrix
from sklearn.metrics import hamming_loss, coverage_error, label_ranking_average_precision_score, label_ranking_loss,f1_score

def hLoss(plabels, test_target):
    loss = hamming_loss(test_target, plabels)
    return(loss)

def rLoss(creds, test_target):
    loss = label_ranking_loss(test_target, creds)
    return(loss)

def oError(creds, test_target):
    
    y_score = creds
    y_true = test_target

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

    return(OneError)

def Coverage(creds, test_target):
    
    n_labels = test_target.shape[1] * 1.0
    loss = coverage_error(test_target,creds)

    return((loss - 1) / n_labels)

def avgPre(creds, test_target):

    avgPrecision = label_ranking_average_precision_score(test_target, creds)

    return(avgPrecision)

def macroF1(plabels,test_target):
    
    macrof1 = f1_score(test_target,plabels,average="macro")
    
    return(macrof1)

def microF1(plabels,test_target):
    
    microf1 = f1_score(test_target,plabels,average="micro")
    
    return(microf1)

def exactMatch(plabels,test_target):
    
    num_ins,num_lab = np.shape(test_target)
    loss = 0
    for i in range(num_ins):
        if (plabels[i]==test_target[i]).all():
            loss += 1
    
    return(loss/num_ins)

def eval_and_print(creds,plabels,test_target):
    a = hLoss(plabels,test_target)
    b = rLoss(creds, test_target)
    c = oError(creds, test_target)
    d = Coverage(creds, test_target)
    e = avgPre(creds, test_target)
    print(f"hamming_loss={str(a)}, ranking_loss={str(b)}, oneerror={str(c)}, coverage={str(d)}, average_precision={str(e)}\n")