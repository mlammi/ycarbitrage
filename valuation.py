"""
In this .py file valuation is implemented:

-Bootstrapping OIS discount factors
-Continuous OIS zero curve with cubic interpolation
-Continuous swap curve with linear interpolation
-Continuous LIBOR curvw with linear interpolation
-Implied forward rates based on the curves above
-Helper functions for e.g. transforming 6 month rates to corresponding 3 month rates
-Valuation of fixed leg in swap
-Valuation of floating leg in swap
(-DV01, duration)

All formulas are based on monthly data and different time intervals are expressed as months.
The formulas are based on the paper of Smith (2013): Valuing Interest Rate Swaps Using Overnight Indexed Swap (OIS) Discounting, The Journal of Derivatives
"""

# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Importing necessary packages/modules


import math
import pandas as pd
import numpy as np
from scipy import interpolate
from dateutil.relativedelta import relativedelta


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Deriving the necessary curves


# Solving OIS discount factors in a month. Inputs: Series (date, 1 week, 2 weeks, 3 weeks, 1 month OIS, 2 month OIS...11 months OIS, 1 year OIS, 2 year OIS...10 year OIS) / Date for which OIS dfs are solved / Day count convention of OIS rates
def bootstrap_ois_month(ois_data, date, convention_ois):
    discount_factors = [date]

    # Calculating 1 week, 2 weeks and 3 weeks OIS discount factor separately and adding them to the discount_factors list. Simplifying assumption that 1 day (overnight) is 1/30 of a month
    df_ois_1w = 1/(1 + ois_data[1] * aj(date, 0, 7/30, convention_ois))
    df_ois_2w = 1/(1 + ois_data[2] * aj(date, 0, 14/30, convention_ois))
    df_ois_3w = 1/(1 + ois_data[3] * aj(date, 0, 21/30, convention_ois))
    discount_factors.append(df_ois_1w)
    discount_factors.append(df_ois_2w)
    discount_factors.append(df_ois_3w)

    # Calculating the OIS discount factors for 1 - 12 months, i.e. less or equal to a year. Simple interest applied and OIS fixed leg payment only at maturity
    for i in range(4, 16):
        discount_factors.append(1/(1 + ois_data[i]*aj(date, 0, i-3, convention_ois)))

    # Calculating the OIS discount factors for 13 - 24 months (2 years). OIS fixed leg assumed to give payments in intervals of 1 year
    for j in range(16, 28):
        t1 = j - 15
        t2 = j - 3
        denominator = 1 + ois_data[j] * aj(date, t1, t2, convention_ois)
        nominator = 1 - ois_data[j] * aj(date, 0, t1, convention_ois) * discount_factors[t1+3]
        discount_factors.append(nominator/denominator)

    # Calculating the OIS discount factors for 27 months, 30 months, 33 months and 36 months (3 years), CHECK
        temp = 0
    for m in range(28, 32):
        # temp = 0
        t1 = (m+temp) - 25
        t2 = (m+temp) - 13
        t3 = (m+temp) - 1
        denominator = 1 + ois_data[m] * aj(date, t2, t3, convention_ois)
        nominator = 1 - (ois_data[m] * aj(date, 0, t1, convention_ois) * discount_factors[t1+3] + ois_data[m] * aj(date, t1, t2, convention_ois) * discount_factors[t2+3])
        temp += 2
        discount_factors.append(nominator/denominator)

    # Calculating the OIS discount factors for 39 months, 42 months, 45 months and 48 months (4 years)
        temp = 7
    for n in range(32, 36):
        # temp = 7
        t1 = (n+temp) - 36
        t2 = (n+temp) - 24
        t3 = (n+temp) - 12
        t4 = (n+temp)
        denominator = 1 + ois_data[n] * aj(date, t3, t4, convention_ois)
        nominator = 1 - (ois_data[n] * aj(date, 0, t1, convention_ois) * discount_factors[t1+3] + ois_data[n] * aj(date, t1, t2, convention_ois) * discount_factors[t2+3] + ois_data[n] * aj(date, t2, t3, convention_ois) * discount_factors[n-4])
        temp += 2
        discount_factors.append(nominator/denominator)

    # Calculating the OIS discount factors for 51 months, 54 months, 57 months and 60 months (5 years)
        temp = 15
    for z in range(36, 40):
        # temp = 15
        t1 = (z+temp) - 48
        t2 = (z+temp) - 36
        t3 = (z+temp) - 24
        t4 = (z+temp) - 12
        t5 = (z+temp)
        denominator = 1 + ois_data[z] * aj(date, t4, t5, convention_ois)
        nominator = 1 - (ois_data[z] * aj(date, 0, t1, convention_ois) * discount_factors[t1+3] + ois_data[z] * aj(date, t1, t2, convention_ois) * discount_factors[t2+3] + ois_data[z] * aj(date, t2, t3, convention_ois) * discount_factors[z-8] + ois_data[z] * aj(date, t3, t4, convention_ois) * discount_factors[z-4])
        temp += 2
        discount_factors.append(nominator/denominator)

    # Calculating the OIS discount factors for 6, 7, 8, 9 and 10 years
    year_disc_facs = []
    year_disc_facs.append(discount_factors[15])
    year_disc_facs.append(discount_factors[27])
    year_disc_facs.append(discount_factors[31])
    year_disc_facs.append(discount_factors[35])
    year_disc_facs.append(discount_factors[39])

    for t in range(40, 45):
        temp = 0
        sum_of_ajdf = 0
        for disc_fac in year_disc_facs:
            sum_of_ajdf += disc_fac * aj(date, temp*12, (temp+1)*12, convention_ois)
            temp += 1
        nominator = 1 - ois_data[t] * sum_of_ajdf
        denominator = 1 + ois_data[t] * aj(date, temp*12, (temp+1)*12, convention_ois)
        df_ois = nominator / denominator
        # temp = 0
        discount_factors.append(df_ois)
        year_disc_facs.append(df_ois)

    return discount_factors  # Output: List of discount factors


# Solving OIS discount factors for full dataset, applying the bootstrap_ois_month - function. Inputs: Dataframe of OIS rates / Day count convention of OIS rates
def bootstrap_ois(ois_data, ois_convention):
    discount_factors = []
    for index, date_data in ois_data.iterrows():
        discount_factors.append(bootstrap_ois_month(date_data, date_data[0], ois_convention))
    return pd.DataFrame(discount_factors, columns=['date', 'SWOIS', '2WOIS', '3WOIS', '1MOIS', '2MOIS', '3MOIS', '4MOIS', '5MOIS', '6MOIS', '7MOIS', '8MOIS', '9MOIS', '10MOIS', '11MOIS',
                                                     '1YOIS', '13MOIS', '14MOIS', '15MOIS', '16MOIS', '17MOIS', '18MOIS', '19MOIS', '20MOIS', '21MOIS', '22MOIS', '23MOIS', '2YOIS',
                                                    '27MOIS', '30MOIS', '33MOIS', '3YOIS',
                                                    '39MOIS', '42MOIS', '45MOIS', '4YOIS',
                                                    '51MOIS', '54MOIS', '57MOIS', '5YOIS',
                                                    '6YOIS', '7YOIS', '8YOIS', '9YOIS', '10YOIS'])  # Output: Dataframe of OIS discount factors


# Solving OIS discount factors for all monthly maturities by cubic spline interpolating the bootstrapped OIS discount factors. Input: Dataframe of bootstrapped OIS discount factors
def cubic_ois(bootstrap_ois_data):

    # Implementing the cubic spline interpolation
    x = [7/30, 14/30, 21/30, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
         27, 30, 33, 36, 39, 42, 45, 48, 51, 54, 57, 60, 72, 84, 96, 108, 120]  # "x-values" of provided data relative to a month.
    x_new = np.linspace(1, 120, num=120)  # "x-values" for the interpolation
    interp_disc_curve = []
    for index, date_data in bootstrap_ois_data.iterrows():
        disc_facs = interpolate.CubicSpline(x, date_data['SWOIS':'10YOIS'].values, bc_type='natural')
        y_new = disc_facs(x_new)
        interp_disc_curve.append(y_new)

    # Setting up the dataframe to be returned
    cubic_index = ['1MOIS', '2MOIS', '3MOIS', '4MOIS', '5MOIS', '6MOIS', '7MOIS', '8MOIS', '9MOIS', '10MOIS',
                   '11MOIS', '12MOIS', '13MOIS', '14MOIS', '15MOIS', '16MOIS', '17MOIS', '18MOIS', '19MOIS', '20MOIS',
                   '21MOIS', '22MOIS', '23MOIS', '24MOIS', '25MOIS', '26MOIS', '27MOIS', '28MOIS', '29MOIS', '30MOIS',
                   '31MOIS', '32MOIS', '33MOIS', '34MOIS', '35MOIS', '36MOIS', '37MOIS', '38MOIS', '39MOIS', '40MOIS',
                   '41MOIS', '42MOIS', '43MOIS', '44MOIS', '45MOIS', '46MOIS', '47MOIS', '48MOIS', '49MOIS', '50MOIS',
                   '51MOIS', '52MOIS', '53MOIS', '54MOIS', '55MOIS', '56MOIS', '57MOIS', '58MOIS', '59MOIS', '60MOIS',
                   '61MOIS', '62MOIS', '63MOIS', '64MOIS', '65MOIS', '66MOIS', '67MOIS', '68MOIS', '69MOIS', '70MOIS',
                   '71MOIS', '72MOIS', '73MOIS', '74MOIS', '75MOIS', '76MOIS', '77MOIS', '78MOIS', '79MOIS', '80MOIS',
                   '81MOIS', '82MOIS', '83MOIS', '84MOIS', '85MOIS', '86MOIS', '87MOIS', '88MOIS', '89MOIS', '90MOIS',
                   '91MOIS', '92MOIS', '93MOIS', '94MOIS', '95MOIS', '96MOIS', '97MOIS', '98MOIS', '99MOIS', '100MOIS',
                   '101MOIS', '102MOIS', '103MOIS', '104MOIS', '105MOIS', '106MOIS', '107MOIS', '108MOIS', '109MOIS', '110MOIS',
                   '111MOIS', '112MOIS', '113MOIS', '114MOIS', '115MOIS', '116MOIS', '117MOIS', '118MOIS', '119MOIS', '120MOIS']
    cub_vals = pd.DataFrame(interp_disc_curve, columns=cubic_index)
    dates = bootstrap_ois_data['date']
    cub_vals.insert(0, 'date', dates, True)
    return cub_vals  # Output: Dataframe of interpolated OIS discount factors


# Solving swap data for all monthly maturities by cubic spline interpolating the swap data. Input: Dataframe of swap rates
def cubic_swap(swap_data):

    # Implementing the cubic spline interpolation
    x = [12, 24, 36, 48, 60, 72, 84, 96, 108, 120]  # "x-values" of provided data relative to a month
    x_new = np.linspace(12, 120, num=109)  # "x-values" for the interpolation
    interp_swaps = []
    for index, date_data in swap_data.iterrows():
        swaps = interpolate.CubicSpline(x, date_data['USD1YS':'USD10YS'].values, bc_type='natural')
        y_new = swaps(x_new)
        interp_swaps.append(y_new)

    # Setting up the dataframe to be returned
    swap_index = ['12MS', '13MS', '14MS', '15MS', '16MS', '17MS', '18MS', '19MS', '20MS',
                  '21MS', '22MS', '23MS', '24MS', '25MS', '26MS', '27MS', '28MS', '29MS', '30MS',
                  '31MS', '32MS', '33MS', '34MS', '35MS', '36MS', '37MS', '38MS', '39MS', '40MS',
                  '41MS', '42MS', '43MS', '44MS', '45MS', '46MS', '47MS', '48MS', '49MS', '50MS',
                  '51MS', '52MS', '53MS', '54MS', '55MS', '56MS', '57MS', '58MS', '59MS', '60MS',
                  '61MS', '62MS', '63MS', '64MS', '65MS', '66MS', '67MS', '68MS', '69MS', '70MS',
                  '71MS', '72MS', '73MS', '74MS', '75MS', '76MS', '77MS', '78MS', '79MS', '80MS',
                  '81MS', '82MS', '83MS', '84MS', '85MS', '86MS', '87MS', '88MS', '89MS', '90MS',
                  '91MS', '92MS', '93MS', '94MS', '95MS', '96MS', '97MS', '98MS', '99MS', '100MS',
                  '101MS', '102MS', '103MS', '104MS', '105MS', '106MS', '107MS', '108MS', '109MS', '110MS',
                  '111MS', '112MS', '113MS', '114MS', '115MS', '116MS', '117MS', '118MS', '119MS', '120MS']
    dates = swap_data['date']
    cub_vals = pd.DataFrame(interp_swaps, columns=swap_index)
    cub_vals.insert(0, 'date', dates, True)
    return cub_vals  # Output: Dataframe of interpolated swap rates


# Solving LIBOR data for all monthly maturities by cubic spline interpolating the libor data. Input: Dataframe of LIBOR rates
def cubic_libor(libor_data):

    # Implementing the cubic spline interpolation
    x = [1/30, 7/30, 1, 2, 3, 6, 12]  # "x-values" of provided data relative to a month. Here again simplifying assumptions that overnight is 1/30 of a month and 1 week is 7/30 of a month
    x_new = np.linspace(1, 12, num=12)  # "x-values" for the interpolation
    interp_libor = []
    dates = libor_data['date']
    for index, date_data in libor_data.iterrows():
        swaps = interpolate.CubicSpline(x, date_data['USDONL':'USD12ML'].values, bc_type='natural')
        y_new = swaps(x_new)
        interp_libor.append(y_new)

    # Setting up the dataframe to be returned
    cubic_index = ['1ML', '2ML', '3ML', '4ML', '5ML', '6ML', '7ML', '8ML', '9ML', '10ML', '11ML', '12ML']
    cub_vals = pd.DataFrame(interp_libor, columns=cubic_index)
    cub_vals.insert(0, 'date', dates, True)
    return cub_vals  # Output: Dataframe of interpolated LIBOR rates


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Helper functions


# Calculate the number of days between the given date and date + n_months. Inputs: Date / Integer or float depicting the number of months
def days_between(date, n_months):
    date_n_months = add_months(date, n_months)
    diff = date_n_months - date
    return diff.days  # Output: Integer of number of days between the two dates


# Get the date that is n_months after the provided date. If the provided n_months is less than 1, i.e. less than 1 month, convert it to days and add it to the date. Inputs: Date / Integer or float depicting the number of months
def add_months(date, n_months):
    if abs(n_months) < 1:
        n_days = n_months * 30  # Here again a simplifying assumption that 1 month holds 30 days
        date_n_months = date + relativedelta(days=n_days)
        return date_n_months
    date_n_months = date + relativedelta(months=+n_months)
    return date_n_months  # Output: Date


# Change 6-month swap rates to 3-month rates (equal annualized returns). Input: Float (semiannual rate)
def month6_month3(sa_rate):
    sa_ret = math.pow(1 + sa_rate/2, 2)
    root_4_sa = math.pow(sa_ret, 0.25)
    r_q = 4 * (root_4_sa - 1)
    return r_q  # Output: Float (quarterly rate)


# Sum the products of day count fractions and discount factors from the given date to maturity. Inputs: Date / Integer for time to first payment (months) / Integer for time to maturity (months) / Array of discount factor data
# / Integer for step size in months / Day count convention
def sum_of_aj_df(date, time_to_first, maturity, interpol_ois_df_data, step, convention):
    sum_of_aj_dfs = aj(date, 0, time_to_first, convention) * interpol_ois_df_data[time_to_first]  # Product of day count fraction to first payment and the corresponding discount factor is calculated separately

    if time_to_first == maturity:
        return sum_of_aj_dfs
    else:
        for i in range(time_to_first + step, maturity + 1, step):  # From the first payment onwards we move in intervals of "step" and add the products to sum_of_aj_dfs
            sum_of_aj_dfs += aj(date, i - step, i, convention) * interpol_ois_df_data[i]
        return sum_of_aj_dfs  # Output: Float (sum of product of day count fractions and discount factors)


# Sum the products of implied forward rates, day count fractions and discount factors from the given date to maturity. Inputs: Date / Integer for time to first payment (months) / Integer for time to maturity (months)
# / Array of implied forward rate data  / Array of discount factor data / Integer for step size in months / Day count convention
def sum_of_ifr_aj_df(date, time_to_first, maturity, ifrs_data, interpol_ois_df_data, step, convention):
    sum_of_ifr_aj_dfs = ifrs_data[0] * aj(date, 0, time_to_first, convention) * interpol_ois_df_data[time_to_first]  # Product of day count fraction to first payment, the corresponding discount factor and ongoing LIBOR is calculated separately
    if time_to_first == maturity:
        return sum_of_ifr_aj_dfs
    else:
        j = 1
        for i in range(time_to_first + step, maturity + 1, step):  # From the first payment onwards we move in intervals of "step" and add the products to sum_of_ifr_aj_dfs
            sum_of_ifr_aj_dfs += ifrs_data[j] * aj(date, i - step, i, convention) * interpol_ois_df_data[i]
            j += 1
        return sum_of_ifr_aj_dfs  # Output: Float (sum of product of implied forward rates, day count fractions and discount factors)


# Calculate the day count fraction between date + start (months from date) and date + end (months from date) with the given convention. Inputs: Date / Start time from date (months) / End time from date (months) / Day count convention
def aj(date, start, end, convention):

    start_date = add_months(date, start)
    days = days_between(start_date, end-start)

    # If day count convention is "30/360", these are needed, else not
    end_date = add_months(date, end)
    years = int((end - start) / 12)
    months = (end - start) - 12 * years
    end_day = end_date.day
    start_day = start_date.day
    if start_day == 31:
        start_day = 30
    if (end_day == 31) and (start_day == 30 or start_day == 31):
        end_day = 30
    days_30_360 = end_day - start_day

    # Solving the day count fractions
    if convention == "30/360":
        day_frac = (360 * years + 30 * months + days_30_360) / 360
    elif convention == "Act/360":
        day_frac = days/360
    elif convention == "Act/365":
        day_frac = days/365

    return day_frac  # Output: Float (day count fraction with the given day count convention)


# ------------------------------------------e-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Implied forward rates


# Solving the implied forward rates for given date
# Inputs: Date / Time to first payment in months / Time to maturity in months / Dataframe of interpolated LIBOR data / Dataframe of interpolated OIS discount factor data / Dataframe of interpolated swap data / Integer for floating leg step size in months
# / Day count conv.
def ifrs(date, time_to_first_float, time_to_first_fixed, maturity, interp_libor_data, interp_ois_df_data, interp_swap_data, step_float, step_fixed, convention_float, convention_fixed):

    # Extracting the LIBOR, OIS and swap data of the date from interpolated data
    libor_date = interp_libor_data[interp_libor_data['date'] == date].values[0]
    ois_date = interp_ois_df_data[interp_ois_df_data['date'] == date].values[0]
    swap_date = interp_swap_data[interp_swap_data['date'] == date].values[0]
    implied_forwards = [libor_date[time_to_first_float]]  # Adding the first ongoing/known LIBOR rate to the implied forwards

    for i in range(time_to_first_float + step_float, maturity + 1, step_float):

        if i < 12:
            nm = libor_date[i] * ois_date[i] * aj(date, 0, i, convention_float) - sum_of_ifr_aj_df(date, time_to_first_float, i - step_float, implied_forwards, ois_date, step_float, convention_float)
            dnm = aj(date, i - step_float, i, convention_float) * ois_date[i]
            ifr = nm/dnm
            implied_forwards.append(ifr)

        else:
            # nm = month6_month3(swap_date[i-11]) * sum_of_aj_df(date, time_to_first, i, ois_date, step_float, convention_fixed) - sum_of_ifr_aj_df(date, time_to_first, i - step_float, implied_forwards, ois_date, step_float, convention_float)
            nm = swap_date[i - 11] * sum_of_aj_df(date, time_to_first_fixed, i, ois_date, step_fixed, convention_fixed) - sum_of_ifr_aj_df(date, time_to_first_float, i - step_float, implied_forwards, ois_date, step_float, convention_float)
            dnm = aj(date, i-step_float, i, convention_float) * ois_date[i]
            ifr = nm/dnm
            implied_forwards.append(ifr)

    return implied_forwards  # Output: List of implied forwards for the given month


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Valuations, clean prices


# Valuating the floating leg in a swap. Inputs: Date of valuation / Ongoing LIBOR (float rate) / Dataframes of interpolated LIBOR, OIS discount factor and swap data / Integers for time to first floating leg payment and maturity / Nominal of the swap
# / Step size in months / Day count convention
def valuate_float(date, first_float, interp_libor_data, interp_ois_df_data, interp_swap_data, time_to_first_float, time_to_first_fixed, maturity, nominal, step_float, step_fixed, convention_float, convention_fixed):
    ifrs_data = ifrs(date, time_to_first_float, time_to_first_fixed, maturity, interp_libor_data, interp_ois_df_data, interp_swap_data, step_float, step_fixed, convention_float, convention_fixed)  # Solving the implied forward rates for the given month
    disc_data = interp_ois_df_data[interp_ois_df_data['date'] == date].values[0]  # Choosing the discount factor data for the given date (month)
    ifr_aj_df = sum_of_ifr_aj_df(date, time_to_first_float, maturity, ifrs_data, disc_data, step_float, convention_float)  # Sum of the products of implied forward rates, day count fractions and discount factors
    price_float = nominal * first_float * aj(date, 0, time_to_first_float, convention_float) * disc_data[time_to_first_float]
    price_float += nominal * (ifr_aj_df - ifrs_data[0] * aj(date, 0, time_to_first_float, convention_float) * disc_data[time_to_first_float]) + nominal * disc_data[maturity]
    return price_float  # Output: Float (price (value) of the floating leg)


# Valuating the fixed leg in a swap. Inputs: Date of valuation / Fixed swap rate / Dataframe of interpolated OIS discount factors / Time to first fixed leg payment in months / Time to maturity in months / Nominal of the swap
# / Step size in months / Day count convention
def valuate_fixed(date, swap_rate, interp_ois_df_data, time_to_first_fixed, maturity, nominal, step_fixed, convention_fixed):
    disc_data = interp_ois_df_data[interp_ois_df_data['date'] == date].values[0]
    aj_df = sum_of_aj_df(date, time_to_first_fixed, maturity, disc_data, step_fixed, convention_fixed)
    price_fixed = nominal * swap_rate * aj_df + nominal * disc_data[maturity]
    return price_fixed  # Output: Float (value (price) of the fixed leg)


# "Clean price" valuating of receiver swap as the difference of values of fixed leg and floating leg. Accrued interest not considered in this value. Inputs: Date of valuation / Ongoing LIBOR (float rate) / Fixed swap rate / Dataframes of interpolated LIBOR,
# OIS discount factor and swap data / Time to first fixed leg payment, floating leg payment and maturity in months / Nominal of the swap / Fixed leg step size and floating leg step size in months / Day count convention of the fixed leg and floating leg
def valuate_receiver_swap_clean(date, float_rate, swap_rate, interp_libor_data, interp_ois_df_data, interp_swap_data, time_to_first_fixed, time_to_first_float, nominal, maturity, fixed_step, float_step, convention_fixed, convention_float):
    fixed_price = valuate_fixed(date, swap_rate, interp_ois_df_data, time_to_first_fixed, maturity, nominal, fixed_step, convention_fixed)
    float_price = valuate_float(date, float_rate, interp_libor_data, interp_ois_df_data, interp_swap_data, time_to_first_float, time_to_first_fixed, maturity, nominal, float_step, fixed_step, convention_float, convention_fixed)
    return fixed_price - float_price  # Output: Float (value of the receiver swap as difference of fixed leg value and floating leg value)


# "Clean price" valuating of receiver swap as the difference of values of fixed leg and floating leg. Accrued interest not considered in this value. Inputs: Date of valuation / Ongoing LIBOR (float rate) / Fixed swap rate / Dataframes of interpolated LIBOR,
# OIS discount factor and swap data / Time to first fixed leg payment, floating leg payment and maturity in months / Nominal of the swap / Fixed leg step size and floating leg step size in months / Day count convention of the fixed leg and floating leg
def valuate_payer_swap_clean(date, float_rate, swap_rate, interp_libor_data, interp_ois_df_data, interp_swap_data, time_to_first_fixed, time_to_first_float, nominal, maturity, fixed_step, float_step, convention_fixed, convention_float):
    fixed_price = valuate_fixed(date, swap_rate, interp_ois_df_data, time_to_first_fixed, maturity, nominal, fixed_step, convention_fixed)
    float_price = valuate_float(date, float_rate, interp_libor_data, interp_ois_df_data, interp_swap_data, time_to_first_float, time_to_first_fixed, maturity, nominal, float_step, fixed_step, convention_float, convention_fixed)
    return float_price - fixed_price  # Output: Float (value of the payer swap as difference of floating leg value and fixed leg value)


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Valuations, dirty prices


# Solving the accrued interest for the floating leg of open swap position. Inputs: Date of valuation / List of open swap position data / Day count convention of the floating leg rate / Type of the trade, i.e. hedging or misvaluation
def accrued_interest_float(date, position_data, convention_float, trade_type, step_float):
    time_to_float = position_data[4]
    year_frac = aj(date, -(step_float - time_to_float), 0, convention_float)
    if trade_type == "main":
        nominal = position_data[15]
    elif trade_type == "h1":
        nominal = position_data[16]
    else:
        nominal = position_data[17]
    latest_libor = position_data[10]
    return nominal * year_frac * latest_libor  # Output: Float (accrued interest as product of swap nominal, day count (year) fraction and ongoing LIBOR (floating leg rate))


# Solving the accrued interest for the fixed leg of open swap position. Inputs: Date of valuation / List of open swap position data / Day count convention of the fixed leg rate / Type of the trade, i.e. hedging or misvaluation
def accrued_interest_fixed(date, position_data, convention_fixed, trade_type, step_fixed):
    time_to_fixed = position_data[5]
    year_frac = aj(date, -(step_fixed - time_to_fixed), 0, convention_fixed)
    if trade_type == "main":
        nominal = position_data[15]
        fixed_rate = position_data[7]
    elif trade_type == "h1":
        nominal = position_data[16]
        fixed_rate = position_data[8]
    else:
        nominal = position_data[17]
        fixed_rate = position_data[9]
    return nominal * year_frac * fixed_rate  # Output: Float (accrued interest as product of swap nominal, day count (year) fraction and fixed swap rate)


# "Dirty price" valuating the swap as sum of clean price value and difference of accrued interest of fixed leg and floating leg. Inputs: Date of valuation / List of open swap position data / Dataframes of interpolated LIBOR data, interpolated OIS discount
# factor data and interpolated swap data / Day count convention of the floating leg of swap / Day count convention of the fixed leg of swap / Floating leg step size in months / Fixed leg step size in months / Type of the trade, i.e. hedging or misvaluation
def valuate_receiver_swap_dirty(date, position_data, interp_libor_data, interp_ois_df_data, interp_swap_data, convention_float, convention_fixed, float_step, fixed_step, trade_type):
    float_rate = position_data[10]
    time_to_first_fixed = position_data[5]
    time_to_first_float = position_data[4]
    maturity = position_data[3] * 12
    months_from_open = position_data[2]
    if trade_type == "main":
        nominal = position_data[15]
        swap_rate = position_data[7]
        maturity = maturity - months_from_open
    elif trade_type == "h1":
        nominal = position_data[16]
        swap_rate = position_data[8]
        maturity = 12 - months_from_open
    else:
        nominal = position_data[17]
        swap_rate = position_data[9]
        maturity = 120 - months_from_open

    clean_price = valuate_receiver_swap_clean(date, float_rate, swap_rate, interp_libor_data, interp_ois_df_data, interp_swap_data, time_to_first_fixed, time_to_first_float, nominal, maturity, fixed_step, float_step, convention_fixed, convention_float)
    accrued_interest = accrued_interest_fixed(date, position_data, convention_fixed, trade_type, fixed_step) - accrued_interest_float(date, position_data, convention_float, trade_type, float_step)
    return clean_price + accrued_interest  # Output: Float


# "Dirty price" valuating the swap as sum of clean price value and difference of accrued interest of fixed leg and floating leg. Inputs: Date of valuation / List of open swap position data / Dataframes of interpolated LIBOR data, interpolated OIS discount
# factor data and interpolated swap data / Day count convention of the floating leg of swap / Day count convention of the fixed leg of swap / Floating leg step size in months / Fixed leg step size in months / Type of the trade, i.e. hedging or misvaluation
def valuate_payer_swap_dirty(date, position_data, interp_libor_data, interp_ois_df_data, interp_swap_data, convention_float, convention_fixed, float_step, fixed_step, trade_type):
    float_rate = position_data[10]
    time_to_first_fixed = position_data[5]
    time_to_first_float = position_data[4]
    maturity = position_data[3] * 12
    months_from_open = position_data[2]
    if trade_type == "main":
        nominal = position_data[15]
        swap_rate = position_data[7]
        maturity = maturity - months_from_open
    elif trade_type == "h1":
        nominal = position_data[16]
        swap_rate = position_data[8]
        maturity = 12 - months_from_open
    else:
        nominal = position_data[17]
        swap_rate = position_data[9]
        maturity = 120 - months_from_open

    clean_price = valuate_payer_swap_clean(date, float_rate, swap_rate, interp_libor_data, interp_ois_df_data, interp_swap_data, time_to_first_fixed, time_to_first_float, nominal, maturity, fixed_step, float_step, convention_fixed, convention_float)
    accrued_interest = accrued_interest_float(date, position_data, convention_float, trade_type, float_step) - accrued_interest_fixed(date, position_data, convention_fixed, trade_type, fixed_step)
    return clean_price + accrued_interest  # Output: Float

