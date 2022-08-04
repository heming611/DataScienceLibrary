def get_next_date(current_date, increment):
    '''
    functionality: get next date for increment days ahead of current_date
    '''
    return_date = (pd.Timestamp(current_date)+pd.Timedelta(days=increment)).date()
    
    return str(return_date)


def add_months(start_date, months):
    '''
    functionality: adding months to current_date and return a new date
    '''
    month = start_date.month - 1 + months
    year = start_date.year + month // 12
    month = month % 12 + 1
    day = min(start_date.day, calendar.monthrange(year, month)[1])
    
    return datetime.date(year, month, day)

def sample_size(mu1, sigma1, mu2, sigma2, ratio, alpha=0.05, beta=0.20):
    """
    functionality: power calculation
    mu1,mu2: estimated means of populations 1 and 2
    sigma1,sigma2: estimated standard deviation of populations 1 and 2
    alpha: significance level (type I error)
    beta: type II error, (1-beta) is the statistical power
    ratio: estimated ratio of sample size 2 over sample size 1 in the experiment
    """  
    output = {}
    z_alpha = norm.ppf(1-alpha/2)
    z_beta = norm.ppf(1-beta)
    n1 = (pow(sigma1,2)+pow(sigma2,2)/ratio)*(z_alpha+z_beta)**2/((mu2-mu1)**2)
    n2 = ratio*n1
    output['sample size 1'] = int(n1)
    output['sample size 2'] = int(n2)
    
    return output


from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve, average_precision_score


def get_date_in_days(current_date, days):
    '''
    functionality: return date days from current_date
    current_date: String
    days: int
    return: String
    '''
    return str((pd.Timestamp(current_date)+pd.Timedelta(days=days)).date())

    
def normalized_binary_confusion_matrix(label, positive_probs, threshold = 0.5):
    """
    functionality: return normalized confusion matrix, normalized by the total count
    label: ground truth
    positive_probs: predicted positive probability vector
    
    """
    y_pred = (np.array(positive_probs)>threshold).astype(int)
    cnf_matrix = confusion_matrix(label, y_pred)
    total = cnf_matrix.sum() 
    tn_percent, fp_percent, fn_percent, tp_percent = cnf_matrix.ravel()/total
    cnf_matrix_normalized = pd.DataFrame([(tn_percent, fp_percent),
                                         (fn_percent, tp_percent)],
                                        index=['- (actual)', '+ (actual)'],
                                        columns=('- (predicted)', '+ (predicted)'))

    print("normalized confusion matrix")
    print(cnf_matrix_normalized)
    
    return #cnf_matrix_normalized


def print_progress(index, total_length, num_of_dots = 10):
    '''
    functionality: print progress bar in terms of dots
    index: the index: 0 to n
    total_length: length of the for loop
    num_of_dots: number of dots in the progress bar
    '''
    _ = total_length//num_of_dots
    if index % _ == 0:
        print(".", end="")
        
    return
