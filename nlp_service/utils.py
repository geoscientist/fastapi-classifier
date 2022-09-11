import pandas as pd
import numpy as np
from sklearn.metrics import precision_score, recall_score, classification_report


def build_final_dataset(dataset, predictions, predictions_probability, class_labels, cls_prob):

    for idx, cl in enumerate(class_labels, 0):
        dataset[cl+'_prediction'] = predictions[idx]
        dataset[cl+'_prediction_proba'] = predictions_probability[idx][:,1]
        
    for cl in class_labels:
        for idx in range(dataset.shape[0]):
            dataset.at[idx,cl+'_prediction'] = int(dataset.loc[idx,cl+'_prediction_proba'] > cls_prob[cl]) * dataset.loc[idx,cl+'_prediction']
    
    dataset['num_of_classes_predicted'] = sum([dataset[cl+'_prediction'] for cl in class_labels])
    dataset['total_prediction'] = ''

    for cl in class_labels:
        dataset['total_prediction'] += dataset[cl+'_prediction'].apply(lambda x: cl*x + ',')
    
    dataset['total_prediction'] = dataset['total_prediction'].apply(lambda x: x.split(','))
    dataset['total_prediction'] = dataset['total_prediction'].apply(lambda x: ','.join([elem for elem in x if len(elem) > 0]))
    dataset['total_prediction'] = dataset['total_prediction'].apply(lambda x: 'empty' if len(x) == 0 else x)

    return dataset
        

def classification_report_multiclass(true_values, predicted_values):
    classes = list(set([val for val in true_values.unique() if ',' not in val]))
    print(f'CLASSES:{classes}\n')
    
    max_l = max([len(cl) for cl in classes])
    
    results = {}
    
    for cl in classes:
        cur_true = true_values.str.contains(cl).values
        cur_pred = predicted_values.str.contains(cl).values
        tp = 0
        tn = 0
        fp = 0
        fn = 0
        for i in range(len(cur_true)):
            if cur_true[i] == 1:
                if cur_pred[i] == 1:
                    tp += 1
                else:
                    fn += 1
            else:
                if cur_pred[i] == 1:
                    fp += 1
                else:
                    tn += 1
                    
        results[cl] = [0, 0]
        if (tp+fp) != 0:
            results[cl][0] = tp/(tp+fp)
            
        if (tp+fn) != 0:
            results[cl][1] = tp/(tp+fn)
            
        whitespace = ' '
                
        
        print(f'{cl.upper()}{whitespace*(max_l - len(cl))} --- Precision: {results[cl][0]},{whitespace*(20 - len(str(results[cl][0])))} Recall: {results[cl][1]}')
    prec = precision_score(true_values, predicted_values, average=None)
    rec = recall_score(true_values, predicted_values, average=None)
    print(f"total precision {prec}")
    print(f"total recall {rec}")
    print(classification_report(true_values, predicted_values))