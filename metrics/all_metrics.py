import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from math import sqrt

def BinaryClassificationMetrics(loader : DataLoader , model : nn.Module)->dict:
    """
    Test you Binary classification model against comprehensive list of metrics , Includes :
    - accuracy
    - precision
    - recall
    - F1 score
    - specificity
    - AUC-ROC
    - Kappa
    - MCC

    Args:
    loader : Test Dataloader function
    model : Model to be tested

    Returns:
    Metrics
    """
    model.cuda()
    model.eval()

    #accuracy
    num_correct = 0
    num_samples = 0

    #precision
    true_positive = 0
    false_positive = 0

    #recall
    false_negative = 0
    #specificity
    true_negative = 0

    all_probs = []
    all_labels = []

    with torch.no_grad():
        for img , label in loader:

            img = img.cuda()
            label = label.cuda()

            out = model(img)
            probab = torch.softmax(out , dim=1)
            _ , pred = out.max(1)
            
            all_probs.append(probab[: , 1].cpu())
            all_labels.append(label.cpu())
            #accuracy
            num_correct += (pred == label).sum()
            num_samples += pred.size(0)

            #precision
            true_positive += ((pred == 1) & (label==1)).sum().item()
            false_positive += ((pred == 1) & (label == 0)).sum().item()

            #recall
            false_negative += ((pred == 0) & (label == 1)).sum().item()

            true_negative += ((pred == 0) & (label == 0)).sum().item()

    accuracy = (float(num_correct) / float(num_samples)) * 100

    tp_fp = (true_positive + false_positive) 
    precision = (true_positive / tp_fp) if tp_fp > 0 else 0

    tp_fn = (true_positive + false_negative)
    recall = (true_positive / tp_fn) if tp_fn > 0 else 0

    f1_num = 2 * precision * recall
    f1_denom = precision + recall
    f1_score = (f1_num / f1_denom) if f1_denom is not 0 else 0

    tn_fp = (true_negative + false_positive)
    specificity = true_negative / tn_fp if tn_fp > 0 else 0

    #mcc
    num = (true_positive * true_negative) - (false_positive * false_negative)
    denom = sqrt((true_positive + false_positive) * (true_positive + false_negative) * 
                 (true_negative + false_positive) * (true_negative + false_negative))
    mcc = num / denom if denom > 0 else 0

    #kappa coeff
    obs_acc = (true_positive + true_negative) / num_samples
    exp_acc = (((true_positive + false_positive) * 
                (true_positive + false_negative)) + 
                ((false_negative + true_negative) * 
                 (false_positive + true_negative)))  / (num_samples ** 2)


    kappa = (obs_acc - exp_acc) / (1 - exp_acc) if exp_acc != 1 else 0

    try:
        from sklearn.metrics import roc_auc_score

        all_probs = torch.cat(all_probs , dim=0).numpy()
        all_labels = torch.cat(all_labels , dim=0).numpy()

        roc_auc = roc_auc_score(all_labels , all_probs)
    
    except:
        roc_auc = None


    metrics = {
        'accuracy' : accuracy,
        'precision' : precision,
        'recall': recall,
        'specificity': specificity,
        'f1_score': f1_score,
        'mcc': mcc,
        'kappa': kappa,
        'roc_auc': roc_auc,
    }

    return metrics

            

def MultiClassificationMetrics(loader : DataLoader , num_class: int ,model : nn.Module)->dict:
    """
    Test you Multi classification model against comprehensive list of metrics , Includes :
    - accuracy
    - precision
    - recall
    - F1 score
    - specificity
    - AUC-ROC
    

    Args:
        loader (Dataloader): Test Dataloader function
        model (nn.Module): Model to be tested

    Returns:
        Metrics 
    """
    model.cuda()
    model.eval()

    #accuracy
    num_correct = 0
    num_samples = 0

    #for each class
    true_positive = torch.zeros(num_class).cuda()
    false_positive = torch.zeros(num_class).cuda()
    false_negative = torch.zeros(num_class).cuda()
    true_negative = torch.zeros(num_class).cuda()

    all_pred = []
    all_label = []

    with torch.no_grad():
        for img , label in loader:

            img = img.cuda()
            label = label.cuda()

            out = model(img)
            probab = torch.softmax(out , dim=1)
            _ , pred = out.max(1)

            all_pred.append(probab.cpu())
            all_label.append(label.cpu())
            
            #accuracy
            num_correct += (pred == label).sum()
            num_samples += pred.size(0)

            for i in range(num_class):
                #precision
                true_positive[i] += ((pred == i) & (label==i)).sum().item()
                false_positive[i] += ((pred == i) & (label != i)).sum().item()
                #recall
                false_negative[i] += ((pred != i) & (label == i)).sum().item()
                true_negative[i] += ((pred != i) & (label != i)).sum().item()

    accuracy = (float(num_correct) / float(num_samples)) * 100

    #for each class
    precision = torch.zeros(num_class)
    recall = torch.zeros(num_class)
    specificity = torch.zeros(num_class)
    f1_score = torch.zeros(num_class)

    for i in range(num_class):
        precision[i] = true_positive[i] / (true_positive[i] + false_positive[i]) if (true_positive[i] + false_positive[i]) > 0 else 0
        recall[i] = true_positive[i] / (true_positive[i] + false_negative[i]) if (true_positive[i] + false_negative[i]) > 0 else 0
        specificity[i] = true_negative[i] / (true_negative[i] + false_positive[i]) if (true_negative[i] + false_positive[i]) > 0 else 0
        f1_score[i] = (2 * precision[i] * recall[i]) / (precision[i] + recall[i]) if (precision[i] + recall[i]) > 0 else 0

    
    precision_avg = precision.mean().item()
    recall_avg = recall.mean().item()
    f1_score_avg = f1_score.mean().item()
    specificity_avg = specificity.mean().item()


    all_pred = torch.cat(all_pred , dim=0).numpy()
    all_label = torch.cat(all_label , dim=0).numpy()


    try:
        from sklearn.metrics import roc_auc_score

        roc_auc = roc_auc_score(
            np.eye(num_class)[all_label],
            all_pred,
            multi_class='ovr',
            average='macro'
            

        )
    except:
        roc_auc = None

    metrics = {
        'accuracy' : accuracy,
        'per_class_precision' : precision.tolist(),
        'per_class_recall' : recall.tolist(),
        'per_class_f1' : f1_score.tolist(),
        'per_class_specificity' : specificity.tolist(),
        'precision_avg' : precision_avg,
        'recall_avg' : recall_avg,
        'f1_avg' : f1_score_avg,
        'specificity_avg' : specificity_avg,
        'roc_auc' : roc_auc
    }

    return metrics

    



            

    