"""
Implementing the short rate (term structure) Vasicek model with 2 factors x and y. The Vasicek model referred to
is the Vasicek 2-factor model without shift extension. It is not as widely recognized as the
no-arbitrage version, that incorporates the shift extension.
For notation, please refer to the book 'Interest Rate Models - Theory and Practice' (2006) by Damiano & Mercurio.

Other notes:
-t1 is the starting time and t2 is maturity. For current moments t1 is 0. For forward starting swaps t1 > 0
-step refers to the payment schedule intervals and is to be provided as per year step, i.e. 0.25 years for 3 months etc
-Check the repository data for indexes and columns
"""

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Importing necessary packages/modules


import math
import pandas as pd
import numpy as np
from scipy import optimize
from sklearn.metrics import mean_squared_error
from pyomo.environ import *
from scipy.special import logsumexp
import random
import pyomo.contrib.parmest.parmest as parmest


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Implementing the zero coupon bond pricing by Vasicek 2-factor model


# Helper function for zero coupon bond pricing and solving partial derivatives of swap rates w.r.t. Vasicek factors - Float
def b(t1, t2, k):
    return (1 / k) * (1 - np.exp(-k * (t2 - t1)))
    # return (1 / k) * (1 - exp(-k * (t2 - t1)))


# Helper function for zero coupon bond pricing and solving partial derivatives of swap rates w.r.t. Vasicek factors - Float
def a(t1, t2, k, theta, sigma):
    b_val = b(t1, t2, k)
    b_sq = math.pow(b_val, 2)
    sigma_sq = math.pow(sigma, 2)
    k_sq = math.pow(k, 2)
    return np.exp((theta - sigma_sq / (2 * k_sq)) * (b_val - t2 + t1) - sigma_sq / (4 * k) * b_sq)

    # b_sq = b_val**2
    # sigma_sq = sigma**2
    # k_sq = k**2
    # return exp((theta - sigma_sq / (2 * k_sq)) * (b_val - t2 + t1) - sigma_sq / (4 * k) * b_sq)


# Zero coupon bond pricing when 6 parameters, 2 factors, starting time t1 and maturity t2 are known - Float
def zc_price(k1, theta1, sigma1, k2, theta2, sigma2, x, y, t1, t2):
    e_temp = np.exp(-b(t1, t2, k1) * x - b(t1, t2, k2) * y)
    # e_temp = exp(-b(t1, t2, k1) * x - b(t1, t2, k2) * y)
    return a(t1, t2, k1, theta1, sigma1) * a(t1, t2, k2, theta2, sigma2) * e_temp


# Swap rate based on known 6 parameters, 2 factors, t1, t2 and step - Float
def swap_rate(k1, theta1, sigma1, k2, theta2, sigma2, x, y, t1, t2, step):
    sum_of_zc_prices = 0
    t_i = t1 + step
    while t_i <= t2:
        sum_of_zc_prices += zc_price(k1, theta1, sigma1, k2, theta2, sigma2, x, y, t1, t_i)
        t_i += step
    zc_price_t2 = zc_price(k1, theta1, sigma1, k2, theta2, sigma2, x, y, t1, t2)
    return (1/step) * (1 - zc_price_t2) / sum_of_zc_prices


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Helper functions


# Helper function for choosing the calibration period (previous calib_period_len months), that is used in fitting the Vasicek model to the market data - Dataframe
def calib_period(swap_data, date, calib_period_len):
    date_index = swap_data[swap_data['date'] == date].index
    calib_period_start_index = date_index[0] - calib_period_len
    if calib_period_start_index < 0:
        calib_period_start_index = 0
    # return swap_data[calib_period_start_index:date_index[0]]
    return swap_data[calib_period_start_index:date_index[0]].reset_index(drop=True)


# Helper function for all the dates of data that follow a provided date (inclusive) - Array
def choose_dates(swap_data, date):
    date_data = swap_data['date']
    date_index = date_data[date_data == date].index
    dates = date_data[date_index[0]:].values
    return dates


# Helper function to get 1-year and 10-year swap rates from data for a given date - List
def find_s1_s10_rates(data, date):
    date_data = data[data['date'] == date]
    return [date_data['USD1YS'].values[0], date_data['USD10YS'].values[0]]


# Helper function for sum of squared differences between two dataframes - Float
def sum_of_squared(df1, df2):
    return ((df1 - df2) ** 2).values.sum()


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Solving factors x and y


# Solving factors x and y s.t. 1-year and 10-year swaps priced correctly, when 6 parameters, prices of liquid 1-year and 10-year swaps, t1 and step are known - Array
def solve_xy(k1, theta1, sigma1, k2, theta2, sigma2, t1, step, s1, s10):
    def solve_helper(xy):
        s1_temp = swap_rate(k1, theta1, sigma1, k2, theta2, sigma2, xy[0], xy[1], t1, t1 + 1, step)
        s10_temp = swap_rate(k1, theta1, sigma1, k2, theta2, sigma2, xy[0], xy[1], t1, t1 + 10, step)
        return np.array([s1_temp - s1, s10_temp - s10])
    # result = optimize.root(solve_helper, np.array([random.uniform(-2, 2), random.uniform(-2, 2)]))
    result = optimize.root(solve_helper, np.array([0.05, 0.05]))
    return result.x


# Solving factors x and y in a period when (initial) parameters, t1 and market values for the period are provided - List of lists
def xy_in_calib_period(k1, theta1, sigma1, k2, theta2, sigma2, t1, step, calib_period_data):
    xy_list = []
    for index, date_data in calib_period_data.iterrows():
        s1 = date_data[1]
        s10 = date_data[10]
        xy = solve_xy(k1, theta1, sigma1, k2, theta2, sigma2, t1, step, s1, s10)
        xy_list.append([xy[0], xy[1]])
    return xy_list


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Optimizing the 6 Vasicek parameters


# Optimize the 6 parameters in a given date, based on past calib_period_len months, when the calibration period, market swaps and (initial) 6 parameters are provided - Array
def optimize_parameters(k1, theta1, sigma1, k2, theta2, sigma2, step, date, calib_period_len, swap_data):

    k1 = 0.0113727
    theta1 = 0.0835597527412136
    sigma1 = 0.054829
    k2 = 0.4628664
    theta2 = 0.05191692462447047
    sigma2 = 0.0257381

    calibration_period_data = calib_period(swap_data, date, calib_period_len)
    xy_data_in_calib_period = xy_in_calib_period(k1, theta1, sigma1, k2, theta2, sigma2, 0, step, calibration_period_data)

    def optimize_helper(params):
        model_data = swap_rates_in_period(params[0], params[1], params[2], params[3], params[4], params[5], step, xy_data_in_calib_period)
        return mean_squared_error(calibration_period_data.loc[:, 'USD2YS':'USD9YS'], model_data.loc[:, 'USD2YS':'USD9YS'])

    initial_parameters_for_minimize = np.array([k1, theta1, sigma1, k2, theta2, sigma2])
    # print(initial_parameters_for_minimize)
    bnds = [(1e-6, None), (1e-6, 0.15), (1e-6, 0.15), (1e-6, None), (1e-6, 0.15), (1e-6, 0.15)]

    #cons = ({'type': 'ineq', 'fun': lambda x: x[0] - 1e-6},
    #        {'type': 'ineq', 'fun': lambda x: x[1] - 1e-6},
    #        {'type': 'ineq', 'fun': lambda x: x[2] - 1e-6},
    #        {'type': 'ineq', 'fun': lambda x: x[3] - 1e-6},
    #        {'type': 'ineq', 'fun': lambda x: x[4] - 1e-6},
    #        {'type': 'ineq', 'fun': lambda x: x[5] - 1e-6},
    #        {'type': 'ineq', 'fun': lambda x: 0.15 - x[1] - 1e-6},
    #        {'type': 'ineq', 'fun': lambda x: 0.15 - x[2] - 1e-6},
    #        {'type': 'ineq', 'fun': lambda x: 0.15 - x[4] - 1e-6},
    #        {'type': 'ineq', 'fun': lambda x: 0.15 - x[5] - 1e-6})

    '''
    def SSE(params):
        sumofsqs = 0
        for index, row in calibration_period_data.reset_index(drop=True).iterrows():
            # print(index)
            x = xy_data_in_calib_period[index][0]
            y = xy_data_in_calib_period[index][1]
            p1, p2, p3, p4, p5, p6 = params[0], params[1], params[2], params[3], params[4], params[5]
            for i in range(2, 10):
                model_r = swap_rate(p1, p2, p3, p4, p5, p6, x, y, 0, i, 0.5)
                cali_r = row.iloc[i]
                sumofsqs += (model_r-cali_r)**2
        return sumofsqs
    '''

    # optimal_full = optimize.dual_annealing(func=SSE, bounds=bnds)
    optimal_full = optimize.minimize(fun=optimize_helper, x0=initial_parameters_for_minimize, method='L-BFGS-B', bounds=bnds)
    # optimal_full = optimize.dual_annealing(func=optimize_helper, bounds=bnds)
    optimal_par = optimal_full.x
    optimal_val = optimal_full.fun
    optimal_suc = optimal_full.success
    optimal_mes = optimal_full.message
    s = find_s1_s10_rates(swap_data, date)
    xy = solve_xy(optimal_par[0], optimal_par[1], optimal_par[2], optimal_par[3], optimal_par[4], optimal_par[5], 0, step, s[0], s[1])
    return [optimal_par[0], optimal_par[1], optimal_par[2], optimal_par[3], optimal_par[4], optimal_par[5], xy[0], xy[1], optimal_val, optimal_suc, optimal_mes]


# Alternative optimization
def optimize_parameters2(k1, theta1, sigma1, k2, theta2, sigma2, step, date, calib_period_len, swap_data):
    calibration_period_data = calib_period(swap_data, date, calib_period_len)
    xy_data_in_calib_period = xy_in_calib_period(k1, theta1, sigma1, k2, theta2, sigma2, 0, step, calibration_period_data)

    mdl = ConcreteModel()
    mdl.k1 = Var(bounds=(1e-6, None))
    mdl.theta1 = Var(bounds=(1e-6, 0.3))
    mdl.sigma1 = Var(bounds=(1e-6, 0.3))
    mdl.k2 = Var(bounds=(1e-6, None))
    mdl.theta2 = Var(bounds=(1e-6, 0.3))
    mdl.sigma2 = Var(bounds=(1e-6, 0.3))

    def SSE(calidata, xydata, p1, p2, p3, p4, p5, p6):
        sumofsqs = 0
        for index, row in calidata.iterrows():
            # print(index)
            x = xydata[index][0]
            y = xydata[index][1]
            for i in range(2, 10):
                model_r = swap_rate(p1, p2, p3, p4, p5, p6, x, y, 0, i, 0.5)
                cali_r = row.iloc[i]
                sumofsqs += (model_r-cali_r)**2
        return sumofsqs


    # model_data = swap_rates_in_period(mdl.k1, mdl.theta1, mdl.sigma1, mdl.k2, mdl.theta2, mdl.sigma2, step, xy_data_in_calib_period)
    # mdl.obj = Objective(expr=sum_of_squared(calibration_period_data.loc[:, 'USD2YS':'USD9YS'], swap_rates_in_period(mdl.k1, mdl.theta1, mdl.sigma1, mdl.k2, mdl.theta2, mdl.sigma2, step, xy_data_in_calib_period).loc[:, 'USD2YS':'USD9YS']), sense=minimize)
    mdl.obj = Objective(expr=SSE(calibration_period_data, xy_data_in_calib_period, mdl.k1, mdl.theta1, mdl.sigma1, mdl.k2, mdl.theta2, mdl.sigma2), sense=minimize)
    SolverFactory('scip', executable=r'C:\Users\Miro_\Desktop\koodit\baron\scipampl-6.0.0.win.x86_64.intel.opt.spx2.exe').solve(mdl)
    # SolverFactory('baron', executable=r'C:\Users\Miro_\Desktop\koodit\baron\baron').solve(mdl)
    # mdl.pprint()

    s = find_s1_s10_rates(swap_data, date)
    xy = solve_xy(mdl.k1(), mdl.theta1(), mdl.sigma1(), mdl.k2(), mdl.theta2(), mdl.sigma2(), 0, step, s[0], s[1])
    optimal_val, optimal_suc, optimal_mes = 'jee', 'jee', 'jee'
    return [mdl.k1(), mdl.theta1(), mdl.sigma1(), mdl.k2(), mdl.theta2(), mdl.sigma2(), xy[0], xy[1], optimal_val, optimal_suc, optimal_mes]


# Optimizing parameters after a date for all dates (inclusive) - Dataframe
def optimal_parameters_after_date(k1, theta1, sigma1, k2, theta2, sigma2, step, date, calib_period_len, swap_data):
    parameters = [k1, theta1, sigma1, k2, theta2, sigma2]
    dates = choose_dates(swap_data, date)
    parameters_of_dates = []
    for date_temp in dates:
        parameters = optimize_parameters(parameters[0], parameters[1], parameters[2], parameters[3], parameters[4], parameters[5], step, date_temp, calib_period_len, swap_data)
        parameters_of_dates.append([date_temp, parameters[0], parameters[1], parameters[2], parameters[3], parameters[4], parameters[5], parameters[6], parameters[7], parameters[8], parameters[9], parameters[10]])
    return pd.DataFrame(parameters_of_dates, columns=['date', 'k1', 'theta1', 'sigma1', 'k2', 'theta2', 'sigma2', 'x', 'y', 'value', 'success', 'message'])


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Solving the swap rates in different periods


# Solving the 1-10 year swap rates in a given date when factors, parameters and step are known - List
def swap_rates_in_date(k1, theta1, sigma1, k2, theta2, sigma2, x, y, step):
    rates = []
    for i in range(1, 11):
        rate = swap_rate(k1, theta1, sigma1, k2, theta2, sigma2, x, y, 0, i, step)
        rates.append(rate)
    return rates


# Solving the 1-10 year swap rates over a period (length of xy-vector) when factors, parameters and step are known - Dataframe
def swap_rates_in_period(k1, theta1, sigma1, k2, theta2, sigma2, step, xy_data):
    rates_of_dates = []
    for xy_pair in xy_data:
        rates = swap_rates_in_date(k1, theta1, sigma1, k2, theta2, sigma2, xy_pair[0], xy_pair[1], step)
        rates_of_dates.append(rates)
    return pd.DataFrame(rates_of_dates, columns=['USD1YS', 'USD2YS', 'USD3YS', 'USD4YS', 'USD5YS', 'USD6YS', 'USD7YS', 'USD8YS', 'USD9YS', 'USD10YS'])


# Solving model rates after a date for all dates - Dataframe
def swap_rates_after_date(k1, theta1, sigma1, k2, theta2, sigma2, step, date, calib_period_len, swap_data):
    parameters = [k1, theta1, sigma1, k2, theta2, sigma2]
    dates = choose_dates(swap_data, date)
    rates_of_months = []
    parameters_of_months = []
    for date_temp in dates:
        parameters = optimize_parameters(parameters[0], parameters[1], parameters[2], parameters[3], parameters[4], parameters[5], step, date_temp, calib_period_len, swap_data)
        rates = swap_rates_in_date(parameters[0], parameters[1], parameters[2], parameters[3], parameters[4], parameters[5], parameters[6], parameters[7], step)
        rates_of_months.append([date_temp, rates[0], rates[1], rates[2], rates[3], rates[4], rates[5], rates[6], rates[7], rates[8], rates[9]])
        parameters_of_months.append([date_temp, parameters[0], parameters[1], parameters[2], parameters[3], parameters[4], parameters[5], parameters[6], parameters[7], parameters[8], parameters[9], parameters[10]])
        print(date_temp)
        print(parameters)
    return pd.DataFrame(rates_of_months, columns=['date', 'USD1YS', 'USD2YS', 'USD3YS', 'USD4YS', 'USD5YS', 'USD6YS', 'USD7YS', 'USD8YS', 'USD9YS', 'USD10YS']), pd.DataFrame(parameters_of_months, columns=['date', 'k1', 'theta1', 'sigma1', 'k2', 'theta2', 'sigma2', 'x', 'y', 'value', 'success', 'message'])


def same_parameters(k1, theta1, sigma1, k2, theta2, sigma2, step, date, swap_data):
    parameters = [k1, theta1, sigma1, k2, theta2, sigma2]
    dates = choose_dates(swap_data, date)
    rates_of_months = []
    parameters_of_months = []
    parameters = optimize_parameters(parameters[0], parameters[1], parameters[2], parameters[3], parameters[4], parameters[5], step, dates[0], 60, swap_data)
    for date_temp in dates:
        s1s10 = find_s1_s10_rates(swap_data, date_temp)
        xy = solve_xy(parameters[0], parameters[1], parameters[2], parameters[3], parameters[4], parameters[5], 0, step, s1s10[0], s1s10[1])
        rates = swap_rates_in_date(parameters[0], parameters[1], parameters[2], parameters[3], parameters[4], parameters[5], xy[0], xy[1], step)
        rates_of_months.append([date_temp, rates[0], rates[1], rates[2], rates[3], rates[4], rates[5], rates[6], rates[7], rates[8], rates[9]])
        parameters_of_months.append([date_temp, parameters[0], parameters[1], parameters[2], parameters[3], parameters[4], parameters[5], xy[0], xy[1]])
        print(date_temp)
    return pd.DataFrame(rates_of_months, columns=['date', 'USD1YS', 'USD2YS', 'USD3YS', 'USD4YS', 'USD5YS', 'USD6YS', 'USD7YS', 'USD8YS', 'USD9YS', 'USD10YS']), pd.DataFrame(parameters_of_months, columns=['date', 'k1', 'theta1', 'sigma1', 'k2', 'theta2', 'sigma2', 'x', 'y'])




'''
# Solving the 1-10 year swap rates in a given date when factors, parameters and step are known - List
def swap_rates_in_date(k1, theta1, sigma1, k2, theta2, sigma2, x, y, step):
    years = range(1, 11)
    rates = threadify(lambda d: swap_rate(k1, theta1, sigma1, k2, theta2, sigma2, x, y, 0, d, step), years)
    return rates


# Solving the 1-10 year swap rates over a period (length of xy-vector) when factors, parameters and step are known - Dataframe
def swap_rates_in_period(k1, theta1, sigma1, k2, theta2, sigma2, step, xy_data):
    rates = threadify(lambda d: swap_rates_in_date(k1, theta1, sigma1, k2, theta2, sigma2, d[0], d[1], step), xy_data)
    df1 = pd.DataFrame(rates, columns=['USD1YS', 'USD2YS', 'USD3YS', 'USD4YS', 'USD5YS', 'USD6YS', 'USD7YS', 'USD8YS', 'USD9YS', 'USD10YS'])
    return df1


# Solving model rates after a date for all dates - Dataframe
def swap_rates_after_date(k1, theta1, sigma1, k2, theta2, sigma2, step, date, calib_period_len, swap_data, trade_dates):
    parameters = [k1, theta1, sigma1, k2, theta2, sigma2]
    # dates = choose_dates(swap_data, date)
    rates_of_months = []
    parameters_of_dates = []
    for date_temp in trade_dates:
        parameters = optimize_parameters(parameters[0], parameters[1], parameters[2], parameters[3], parameters[4], parameters[5], step, date_temp, calib_period_len, swap_data)
        print(date_temp)
        rates = swap_rates_in_date(parameters[0], parameters[1], parameters[2], parameters[3], parameters[4], parameters[5], parameters[6], parameters[7], step)
        rates_of_months.append([date_temp, rates[0], rates[1], rates[2], rates[3], rates[4], rates[5], rates[6], rates[7], rates[8], rates[9]])
        parameters_of_dates.append([date_temp, parameters[0], parameters[1], parameters[2], parameters[3], parameters[4], parameters[5], parameters[6], parameters[7], parameters[8], parameters[9], parameters[10]])
    return pd.DataFrame(rates_of_months, columns=['date', 'USD1YS', 'USD2YS', 'USD3YS', 'USD4YS', 'USD5YS', 'USD6YS', 'USD7YS', 'USD8YS', 'USD9YS', 'USD10YS']), pd.DataFrame(parameters_of_dates, columns=['date', 'k1', 'theta1', 'sigma1', 'k2', 'theta2', 'sigma2', 'x', 'y', 'value', 'success', 'message'])
'''

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Partial derivatives of swap rates w.r.t to the factors x and y, needed in hedging


def ds_x(k1, theta1, sigma1, k2, theta2, sigma2, x, y, t1, t2, step):
    b_val_t2 = b(t1, t2, k1)
    zc_price_t2 = zc_price(k1, theta1, sigma1, k2, theta2, sigma2, x, y, t1, t2)
    t_i = t1 + step
    sum_of_bvals_times_zc_prices = 0
    sum_of_zc_prices = 0
    while t_i <= t2:
        sum_of_bvals_times_zc_prices += b(t1, t_i, k1)*zc_price(k1, theta1, sigma1, k2, theta2, sigma2, x, y, t1, t_i)
        sum_of_zc_prices += zc_price(k1, theta1, sigma1, k2, theta2, sigma2, x, y, t1, t_i)
        t_i += step
    sum_of_zc_prices_sq = math.pow(sum_of_zc_prices, 2)
    nominator = b_val_t2*zc_price_t2*sum_of_zc_prices+(1-zc_price_t2)*sum_of_bvals_times_zc_prices
    return (1/step) * nominator/sum_of_zc_prices_sq


def ds_y(k1, theta1, sigma1, k2, theta2, sigma2, x, y, t1, t2, step):
    b_val_t2 = b(t1, t2, k2)
    zc_price_t2 = zc_price(k1, theta1, sigma1, k2, theta2, sigma2, x, y, t1, t2)
    t_i = t1 + step
    sum_of_bvals_times_zc_prices = 0
    sum_of_zc_prices = 0
    while t_i <= t2:
        sum_of_bvals_times_zc_prices += b(t1, t_i, k2) * zc_price(k1, theta1, sigma1, k2, theta2, sigma2, x, y, t1, t_i)
        sum_of_zc_prices += zc_price(k1, theta1, sigma1, k2, theta2, sigma2, x, y, t1, t_i)
        t_i += step
    sum_of_zc_prices_sq = math.pow(sum_of_zc_prices, 2)
    nominator = b_val_t2 * zc_price_t2 * sum_of_zc_prices + (1 - zc_price_t2) * sum_of_bvals_times_zc_prices
    return (1/step) * nominator / sum_of_zc_prices_sq


def solve_w1w10_for_date(date, parameter_data, step):
    parameters = parameter_data[parameter_data['date'] == date].values[0]
    w1w10 = []
    # x5 = ds_x(parameters[1], parameters[2], parameters[3], parameters[4], parameters[5], parameters[6], parameters[7], parameters[8], 0, 5, step)
    # y5 = ds_y(parameters[1], parameters[2], parameters[3], parameters[4], parameters[5], parameters[6], parameters[7], parameters[8], 0, 5, step)
    # y1 = ds_y(parameters[1], parameters[2], parameters[3], parameters[4], parameters[5], parameters[6], parameters[7], parameters[8], 0, 1, step)
    # y10 = ds_y(parameters[1], parameters[2], parameters[3], parameters[4], parameters[5], parameters[6], parameters[7], parameters[8], 0, 10, step)
    # x1 = ds_x(parameters[1], parameters[2], parameters[3], parameters[4], parameters[5], parameters[6], parameters[7], parameters[8], 0, 1, step)
    # x10 = ds_x(parameters[1], parameters[2], parameters[3], parameters[4], parameters[5], parameters[6], parameters[7], parameters[8], 0, 10, step)

    def solve_helper(weights, *args):
        dstdx = ds_x(args[1], args[2], args[3], args[4], args[5], args[6], args[7], args[8], 0, args[9], step)
        dstdy = ds_y(args[1], args[2], args[3], args[4], args[5], args[6], args[7], args[8], 0, args[9], step)
        ds1dx = ds_x(args[1], args[2], args[3], args[4], args[5], args[6], args[7], args[8], 0, 1, step)
        ds1dy = ds_y(args[1], args[2], args[3], args[4], args[5], args[6], args[7], args[8], 0, 1, step)
        ds10dx = ds_x(args[1], args[2], args[3], args[4], args[5], args[6], args[7], args[8], 0, 10, step)
        ds10dy = ds_y(args[1], args[2], args[3], args[4], args[5], args[6], args[7], args[8], 0, 10, step)
        diff = np.array([dstdx, dstdy]) - weights[0] * np.array([ds1dx, ds1dy]) - weights[1] * np.array([ds10dx, ds10dy])
        return diff

    for t in range(2, 10):
        arguments = (1, parameters[1], parameters[2], parameters[3], parameters[4], parameters[5], parameters[6], parameters[7], parameters[8], t)
        we1we10 = optimize.fsolve(solve_helper, np.array([0.5, 0.5]), args=arguments)
        we_tuple = (we1we10[0], we1we10[1])
        w1w10.append(we_tuple)

    return w1w10


def solve_w1w10_for_data(parameter_data, step):
    dates = parameter_data['date'].values
    weights_for_data = []
    for date in dates:
        weights_for_data.append([date] + solve_w1w10_for_date(date, parameter_data, step))
    weight_df = pd.DataFrame(weights_for_data, columns=['date', 'weights2', 'weights3', 'weights4', 'weights5', 'weights6', 'weights7', 'weights8', 'weights9'])
    return weight_df


