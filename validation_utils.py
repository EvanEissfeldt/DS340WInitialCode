import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import csv
from sklearn.metrics import roc_curve, precision_recall_curve, auc, brier_score_loss
import string
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf

def temporal_validation_table(data):
    years = list(data.year.value_counts(sort=False).keys()) + ['tot']
    values = list(data.year.value_counts(sort=False).values)
    table = pd.DataFrame(columns=years, index=['external','temporal 1', 'temporal 2', 'final'])
    # form table
    table.loc['external'] = values + [sum(values)]
    table.loc['temporal 1'].iloc[5:10] = values[5:9] + [sum(values[5:9])]
    table.loc['temporal 2'].iloc[7:10] = values[7:9] + [sum(values[7:9])]
    table.loc['final'] = values + [sum(values)]
    return table

def table1(data, variables, columns):
    """
    Table 1 - Use this function to generate descriptive statistics table.
    data: Your dataset.
    variables: The variable names corresponding to names of variables in dataframe that you want to include in the table.
               Should be given with indication of continuous or categorical datatype ('cont','cat') in dictionary: 
               variables = {'age':'cont', 'sex':'cat'}.
    columns: The columns indicate the subgroups you want to analyse. Should be given in dictionary with the keys   
             corresponding to the subgroup labels and the values a binary indicator vector (True/False) indicating the 
             observations to be included in the subgroup:
             columns = {'all':[True]*data.shape[0], 'subgroup1':data.grouping_variable=='subgroup1', 'subgroup2':data.grouping_variable=='subgroup2'}.
             For hospital disposition, grouping_variable will the disposition variable and the subgroups 'ICU', 'home' etc. 
    """
    # Initialize data frame for table and index
    index = []
    table = pd.DataFrame(columns=columns.keys())

    # Calculate total subjects and percentages subjects across subgroups
    table = table.append({
        i: str(data.loc[columns[i],:].shape[0])+' ('+str(round(100*data.loc[columns[i],:].shape[0]/data.shape[0],2))+')' for i in columns.keys() 
    }, ignore_index=True)
    # Append N to index vector.
    index.append('N')

    for v in variables.keys():
        if variables[v]=='cont':
            # Calculate mean and standard deviation per continuous variable
            table = table.append({
                i: str(round(np.mean(data.loc[columns[i],v]),2)) + ' (' + str(round(np.std(data.loc[columns[i],v]),2))+')' for i in columns.keys()
            }, ignore_index=True)
            # Append variable name plus (std) to index vector.
            index.append(v+' (std)')
        elif variables[v]=='cat':
            # Calculate total and percentages per category
            tab = pd.DataFrame({
                i: data.loc[columns[i],v].value_counts(dropna=False).apply(lambda j: str(j)+' ('+str(round(100*j/data.loc[columns[i],:].shape[0],2))+')') for i in columns.keys()
            })
            table = table.append(tab, ignore_index=True)
            # Append variable name plust category name plus (%) to index vector.
            [index.append(v + '-' + str(i) + ' (%)') for i in tab.index]
        else: 
            print('Data type of variable', v, 'must be cat or cont.')
    
    # Set table index/rownames
    table.index = index
    return(table)

def bootstrap_AUC(n, y, prob_y, curve='ROC'):
    y.index = range(0,len(y))
    prob_y.index = range(0,len(y))
    n_bootstraps = n
    rng_seed = 42  # control reproducibility
    rng = np.random.RandomState(rng_seed)
    bootstrapped_scores = []
    for i in range(n_bootstraps):
        # bootstrap by sampling with replacement on the prediction indices
        indices = rng.randint(0, prob_y.shape[0], prob_y.shape[0])
        if len(np.unique(y[indices])) < 2:
             # We need at least one positive and one negative sample for ROC AUC
             # to be defined: reject the sample
             continue
        if curve=='ROC':
            x_axis, y_axis, thresholds = roc_curve(y[indices], prob_y[indices])
        elif curve=='PRC':
            y_axis, x_axis, thresholds = precision_recall_curve(y[indices], prob_y[indices])
        score = auc(x_axis, y_axis)
        bootstrapped_scores.append(score)
    sorted_scores = np.array(bootstrapped_scores)
    sorted_scores.sort()
    # Computing the lower and upper bound of the 90% confidence interval
    # You can change the bounds percentiles to 0.025 and 0.975 to get
    # a 95% confidence interval instead.
    confidence_lower = sorted_scores[int(0.05 * len(sorted_scores))]
    confidence_upper = sorted_scores[int(0.95 * len(sorted_scores))]
    return(confidence_lower, confidence_upper)

def performance_table(data, probabilities, y_true):
    table = pd.DataFrame(columns=['AUROC','AUPRC','calibration int','calibration slope',
        'calibration loss'])
    for m in probabilities:
        p = data.loc[probabilities[m],m]
        y = data.loc[probabilities[m],y_true]
        # discrimination
        precision, recall, _ = precision_recall_curve(y, p)
        auroc_CI = bootstrap_AUC(1000, y, p, curve='ROC')
        fpr, tpr, _ = roc_curve(y, p)
        auprc_CI = bootstrap_AUC(1000, y, p, curve='PRC')
        # calibration
        logit = np.log(p/(1-p))
        df = pd.DataFrame(np.transpose([y,logit]),columns=['y','logit'])
        mod_slope = smf.glm('y~logit', data=df, family=sm.families.Binomial()).fit()
        mod_interc = smf.glm('y~1', data=df, offset=logit, family=sm.families.Binomial()).fit()
        # form table
        table = table.append({
            'AUROC': str(round(auc(fpr, tpr),2)) + ' ' + str(tuple(round(i,2) for i in auroc_CI)),
            'AUPRC': str(round(auc(recall, precision),2)) + ' ' + str(tuple(round(i,2) for i in auprc_CI)),
            'calibration int': str(round(mod_interc.params[0],2)) + ' ' + str(tuple(round(i,2) for i in list(np.array(mod_interc.conf_int(alpha=0.05))[0,:]))),
            'calibration slope': str(round(mod_slope.params[1],2)) + ' ' + str(tuple(round(i,2) for i in list(np.array(mod_slope.conf_int(alpha=0.05))[1,:]))),
            'calibration loss': str(round(calibration_loss(y,p),2))
            },ignore_index=True)
    table.index = probabilities.keys()
    return table


def histogram_plot(data, probabilities):
    plt.figure(figsize=(10,10))
    for ind, val in enumerate(probabilities):
        ax = plt.subplot(2,3,ind+1)  
        ax.hist(data.loc[probabilities[val],val], bins=75, alpha=0.5, lw=3, color='b')
        # add y label
        y_label = string.ascii_lowercase[ind]+')'
        ax.yaxis.set_label_coords(-0.15,0.5)
        ax.set_ylabel(y_label, rotation=0, size=12)
        # add x label
        ax.set_xlabel('Probability_'+val, size=12) 

def density_plot(data, probabilities, title, labels):
    """
    Density plot - Use this function to generate a density plot for the predictions from several models.
    probabilities: A dictionary with probability and corresponding condition. For exmample: {'probability 1':[True]*data_shape[0]}. 
    title: Title of the legend.
    labels: Labels of the legend.
    """
    plt.figure(figsize=(10,10))
    for m in probabilities:    
        sns.distplot(data.loc[probabilities[m],m], hist=False, kde=True, 
            kde_kws={'shade':True,'linewidth':1},
            label=m)
    # Plot formatting
    plt.xlim((0,1))
    plt.legend(title=title,labels=labels)
    plt.xlabel('Probability')
    plt.ylabel('Density')

def calibration_loss(y_true, y_prob, bin_size=100):
    y_prob = y_prob.sort_values()
    index = y_prob.index
    y_true = y_true.iloc[index]
    loss = 0.0
    for i in np.arange(0, len(y_prob)-bin_size):
        avg_prob = y_true.iloc[i:i+bin_size].sum()/bin_size
        mean_pred = y_prob.iloc[i:i+bin_size].sum()/bin_size
        loss += np.abs(mean_pred-avg_prob)
    loss /= (len(y_prob)-bin_size)
    return loss

def roc_prc_plot(data, probabilities, y, labels):
    plt.figure(figsize=(10,5))
    no_skill = data[y].value_counts()[1]/len(data[y])

    # ROC curve
    plt.subplot(1,2,1)
    plt.plot([0,1],[0,1], color='black', linestyle='--')
    for m in probabilities:
        fpr, tpr, _ = roc_curve(data.loc[probabilities[m],y], data.loc[probabilities[m],m])
        roc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=labels[m] + ' (area = %0.2f)'%roc)
    plt.ylabel('sensitivity')
    plt.xlabel('1-specificity')
    plt.legend(loc='lower right')
    plt.title('ROC curve')

    # PRC curve
    plt.subplot(1,2,2)
    plt.plot([0,1],[no_skill,no_skill], color='black', linestyle='--')
    for m in probabilities:
        precision, recall, _ = precision_recall_curve(data.loc[probabilities[m],y], data.loc[probabilities[m],m])
        prc = auc(recall, precision)
        plt.plot(recall, precision, label=labels[m] + ' (area = %0.2f'%prc)
    plt.ylabel('precision')
    plt.xlabel('sensitivity')
    plt.legend(loc='upper right')
    plt.title('PRC curve')

def decision_curve(data, probabilities, y, labels, xlim=[0,1]):
    y = data.loc[:,y]
    event_rate = np.mean(y)
    N = data.shape[0]

    # make nb table
    nb = pd.DataFrame(np.arange(0.01,1,0.01),columns=['threshold'])
    nb['treat_all'] = event_rate - (1-event_rate)*nb.threshold/(1-nb.threshold)
    nb['treat_none'] = 0

    # cycling through each predictor and calculating net benefit
    for m in probabilities:
        nb[m] = 0
        p = data.loc[:,m]
        for ind,t in enumerate(nb.threshold):
            tp = np.mean(y.loc[p>=t])*np.sum(p>=t)
            fp = (1-np.mean(y.loc[p>=t])*np.sum(p>=t))
            if np.sum(p>=t)==0:
                tp=fp=0
            nb.iloc[ind,nb.columns.get_indexer([m])] = tp/N-(fp/N)*(t/(1-t))

    # Make plot
    ymax = np.max(np.max(nb.loc[:,nb.columns!='threshold']))
    plt.figure(figsize=(10,6))
    plt.plot(nb.threshold, nb.treat_all)
    plt.plot(nb.threshold, nb.treat_none)
    for m in probabilities:
        plt.plot(nb.threshold, nb.loc[:,m])
    plt.ylim(bottom=-0.01,top=ymax)
    plt.xlim(left=xlim[0],right=xlim[1])
    plt.legend(title='Predictors', labels=['discharge none','discharge all']+labels)
    plt.xlabel('Threshold')
    plt.ylabel('Net benefit')
