import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import pearsonr, spearmanr, kendalltau


def logistic(x, bayta_1, bayta_2, bayta_3, bayta_4, bayta_5):
    return bayta_1 * (0.5 - 1/(1 + np.exp(bayta_2 * (x - bayta_3)))) + bayta_4 * x + bayta_5


def verify_performance(mos, predict_mos):
    # mos:标签
    # predict_mos：预测分数
    predict_mos = np.array(predict_mos).flatten()
    mos = np.array(mos).flatten()
    # initialize the parameters used by the curve fitting function
    beta = [10, 0.1, np.mean(predict_mos), 0.1, 0.1]
    #beta = [15, 0.2, np.median(predict_mos), 0.2, 0.05]

    # fitting a curve using the data
    bayta, _ = curve_fit(logistic, predict_mos, mos, p0=beta, maxfev=1000000000)
    # given a predict_mos value, predict the corresponding mos (ypre) using the fitted curve
    ypre = logistic(predict_mos, *bayta)
    
    #-----------------------------------------------------------------------------------------
    rmse = np.sqrt(np.sum((ypre - mos)**2) / len(mos))  # root mean squared error
    plcc, _ = pearsonr(mos, ypre)  # Pearson linear coefficient
    #rmse = np.sqrt(np.sum((predict_mos - mos)**2) / len(mos))  # root mean squared error
    #plcc, _ = pearsonr(mos, predict_mos)  # Pearson linear coefficient
    #--------------------------------------------------------------------------------------
    srocc, _ = spearmanr(mos, predict_mos)  # Spearman rank-order correlation coefficient
    krocc, _ = kendalltau(mos, predict_mos)  # Kendall's tau
    return srocc, krocc, plcc, rmse