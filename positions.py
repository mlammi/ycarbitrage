"""
In this .py-file, the basic book keeping for yield curve arbitrage strategy is created.
Functions used:
-xx
-xx
-xx
"""

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Importing necessary packages/modules


import pandas as pd
import valuation as vn
import ast


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Creating the bookkeeping


def create_book():
    book_columns = ['date', 'Trade type', 'M from open', 'Maturity', 'M to float', 'M to fixed', 'S of model', 'S of trade', 'S of H1', 'S of H10', 'Latest 3ML', 'Previous total dirty',
                    'Previous trade dirty', 'Previous H1 dirty', 'Previous H10 dirty', 'Nominal trade', 'Nominal H1', 'Nominal H10', 'Weight H1', 'Weight H10', 'Initial mispricing', 'Current mispricing',
                    'Current total dirty', 'Current trade dirty', 'Current H1 dirty', 'Current H10 dirty']
    book_df = pd.DataFrame(None, columns=book_columns)
    return book_df


def create_dict_of_books():
    first_book_df = create_book()
    return {'First book': first_book_df}


def add_to_book(date, book_df, model_data, market_data, libor_data, weight_data, limit, nominal, step_float, step_fixed):
    misprice = choose_largest_mispricing_and_hedge_rates(date, model_data, market_data, limit)
    libor_date = libor_data[libor_data['date2'] == date].values[0][1]
    book_columns = ['date', 'Trade type', 'M from open', 'Maturity', 'M to float', 'M to fixed', 'S of model', 'S of trade', 'S of H1', 'S of H10', 'Latest 3ML', 'Previous total dirty',
                    'Previous trade dirty', 'Previous H1 dirty', 'Previous H10 dirty', 'Nominal trade', 'Nominal H1', 'Nominal H10', 'Weight H1', 'Weight H10', 'Initial mispricing', 'Current mispricing',
                    'Current total dirty', 'Current trade dirty', 'Current H1 dirty', 'Current H10 dirty']
    if misprice is not None:
        n_w = set_nominals_and_weights(date, nominal, weight_data, misprice[0])
        if misprice[5] < 0:
            mis_type = 'Receiver'
        else:
            mis_type = 'Payer'
        data_to_add = [date, mis_type, 0, misprice[0], step_float, step_fixed, misprice[2], misprice[1], misprice[3], misprice[4], libor_date, 0, 0, 0, 0, n_w[0], n_w[1], n_w[2], n_w[3], n_w[4], misprice[5], misprice[5], 0, 0, 0, 0]
        new_book_df = book_df.append(pd.DataFrame([data_to_add], columns=book_columns), ignore_index=True)
        return new_book_df
    else:
        return book_df


def set_nominals_and_weights(date, trade_nominal, weight_data, maturity):
    weight_date = weight_data[weight_data['date2'] == date].values[0][maturity-1]
    # weight_date = ast.literal_eval(weight_date) #REMOVE IF WEIGHT DATA NOT IMPORTED FROM XLSX-FILE!
    w1 = weight_date[0]
    w10 = weight_date[1]
    return trade_nominal, w1 * trade_nominal, w10 * trade_nominal, w1, w10


# Choosing the largest mispricing above given limit in a given date
def choose_largest_mispricing_and_hedge_rates(date, model_data, market_data, limit):
    model_date = model_data[model_data['date2'] == date].values[0]
    market_date = market_data[market_data['date2'] == date].values[0]

    mispricings = model_date[2:10] - market_date[2:10]
    abs_mispricings = [abs(ele) for ele in mispricings]
    max_abs_mispricing = max(abs_mispricings)
    if max_abs_mispricing < limit:
        return None
    max_abs_mispricing_index = abs_mispricings.index(max_abs_mispricing)
    max_mispricing = mispricings[max_abs_mispricing_index]

    max_mispricing_maturity = max_abs_mispricing_index + 2
    market_mispricing_rate = market_date[max_mispricing_maturity]
    model_mispricing_rate = model_date[max_mispricing_maturity]

    h1_rate = market_date[1]
    h10_rate = market_date[10]

    return max_mispricing_maturity, market_mispricing_rate, model_mispricing_rate, h1_rate, h10_rate, max_mispricing


def update_book(book_df, libor_data, market_data, date):
    market_date = market_data[market_data['date2'] == date].values[0]
    new_book_df = book_df
    number_of_rows = len(book_df.index)
    libor_date_3ml = libor_data[libor_data['date2'] == date].values[0][1]
    for i in range(0, number_of_rows):
        new_book_df.iloc[i, 2] = new_book_df.iloc[i, 2] + 1
        new_book_df.iloc[i, 4] = new_book_df.iloc[i, 4] - 1
        new_book_df.iloc[i, 5] = new_book_df.iloc[i, 5] - 1
        mispricing = new_book_df.iloc[i, 6] - market_date[new_book_df.iloc[i, 3]]
        new_book_df.iloc[i, 21] = mispricing
        if new_book_df.iloc[i, 4] <= 0:
            new_book_df.iloc[i, 4] = 3
            new_book_df.iloc[i, 10] = libor_date_3ml
        if new_book_df.iloc[i, 5] <= 0:
            new_book_df.iloc[i, 5] = 6
    return new_book_df


def update_book2(book_df, libor_data, date):
    new_book_df = book_df
    number_of_rows = len(book_df.index)
    libor_date_3ml = libor_data[libor_data['date2'] == date].values[0][1]
    for i in range(0, number_of_rows):
        new_book_df.iloc[i, 2] = new_book_df.iloc[i, 2] + 1
        new_book_df.iloc[i, 4] = new_book_df.iloc[i, 4] - 1
        new_book_df.iloc[i, 5] = new_book_df.iloc[i, 5] - 1
        if new_book_df.iloc[i, 4] <= 0:
            new_book_df.iloc[i, 4] = 3
            new_book_df.iloc[i, 10] = libor_date_3ml
        if new_book_df.iloc[i, 5] <= 0:
            new_book_df.iloc[i, 5] = 6
    return new_book_df


def update_latest_dirty_prices(date, book_df, cub_libor_data, cub_swap_data, cub_ois_data, convention_float, convention_fixed, step_float, step_fixed):
    new_book_df = book_df
    len_of_book_df = len(book_df.index)
    for i in range(0, len_of_book_df):
        pos_data = new_book_df.iloc[i, :].values
        if pos_data[1] == 'Receiver':
            h1 = vn.valuate_payer_swap_dirty(date, pos_data, cub_libor_data, cub_ois_data, cub_swap_data, convention_float, convention_fixed, step_float, step_fixed, "h1")
            h10 = vn.valuate_payer_swap_dirty(date, pos_data, cub_libor_data, cub_ois_data, cub_swap_data, convention_float, convention_fixed, step_float, step_fixed, "h10")
            trade = vn.valuate_receiver_swap_dirty(date, pos_data, cub_libor_data, cub_ois_data, cub_swap_data, convention_float, convention_fixed, step_float, step_fixed, "main")
            new_book_df.iloc[i, 13] = h1
            new_book_df.iloc[i, 14] = h10
            new_book_df.iloc[i, 12] = trade
            new_book_df.iloc[i, 11] = h1 + h10 + trade
        else:
            h1 = vn.valuate_receiver_swap_dirty(date, pos_data, cub_libor_data, cub_ois_data, cub_swap_data, convention_float, convention_fixed, step_float, step_fixed, "h1")
            h10 = vn.valuate_receiver_swap_dirty(date, pos_data, cub_libor_data, cub_ois_data, cub_swap_data, convention_float, convention_fixed, step_float, step_fixed, "h10")
            trade = vn.valuate_payer_swap_dirty(date, pos_data, cub_libor_data, cub_ois_data, cub_swap_data, convention_float, convention_fixed, step_float, step_fixed, "main")
            new_book_df.iloc[i, 13] = h1
            new_book_df.iloc[i, 14] = h10
            new_book_df.iloc[i, 12] = trade
            new_book_df.iloc[i, 11] = h1 + h10 + trade

    return new_book_df


def update_current_dirty_prices(date, book_df, cub_libor_data, cub_swap_data, cub_ois_data, convention_float, convention_fixed, step_float, step_fixed):
    new_book_df = book_df
    len_of_book_df = len(book_df.index)
    for i in range(0, len_of_book_df):
        pos_data = new_book_df.iloc[i, :].values
        m_from_open = pos_data[2]
        if pos_data[1] == 'Receiver':
            if m_from_open == 12:
                # h1 = pos_data[13]
                h1 = 0
            else:
                h1 = vn.valuate_payer_swap_dirty(date, pos_data, cub_libor_data, cub_ois_data, cub_swap_data, convention_float, convention_fixed, step_float, step_fixed, "h1")
            h10 = vn.valuate_payer_swap_dirty(date, pos_data, cub_libor_data, cub_ois_data, cub_swap_data, convention_float, convention_fixed, step_float, step_fixed, "h10")
            trade = vn.valuate_receiver_swap_dirty(date, pos_data, cub_libor_data, cub_ois_data, cub_swap_data, convention_float, convention_fixed, step_float, step_fixed, "main")
            new_book_df.iloc[i, 24] = h1
            new_book_df.iloc[i, 25] = h10
            new_book_df.iloc[i, 23] = trade
            new_book_df.iloc[i, 22] = h1 + h10 + trade
        else:
            if m_from_open == 12:
                # h1 = pos_data[13]
                h1 = 0
            else:
                h1 = vn.valuate_receiver_swap_dirty(date, pos_data, cub_libor_data, cub_ois_data, cub_swap_data, convention_float, convention_fixed, step_float, step_fixed, "h1")
            h10 = vn.valuate_receiver_swap_dirty(date, pos_data, cub_libor_data, cub_ois_data, cub_swap_data, convention_float, convention_fixed, step_float, step_fixed, "h10")
            trade = vn.valuate_payer_swap_dirty(date, pos_data, cub_libor_data, cub_ois_data, cub_swap_data, convention_float, convention_fixed, step_float, step_fixed, "main")
            new_book_df.iloc[i, 24] = h1
            new_book_df.iloc[i, 25] = h10
            new_book_df.iloc[i, 23] = trade
            new_book_df.iloc[i, 22] = h1 + h10 + trade

    return new_book_df


def clean_book_convergence(book_df):
    new_book_df = create_book()
    close_count_convergence = 0
    for index, row in book_df.iterrows():
        if check_convergence(row):
            new_book_df = new_book_df.append(row, ignore_index=True)
        else:
            close_count_convergence += 1
    return new_book_df, close_count_convergence


def check_convergence(position_data):
    if position_data[21] >= -0.0001 and position_data[1] == 'Receiver':
        return False
    elif position_data[21] <= 0.0001 and position_data[1] == 'Payer':
        return False
    else:
        return True


def clean_book_maturity(book_df):
    new_book_df = create_book()
    close_count_maturity = 0
    for index, row in book_df.iterrows():
        if check_maturity(row):
            new_book_df = new_book_df.append(row, ignore_index=True)
        else:
            close_count_maturity += 1
    return new_book_df, close_count_maturity


def check_maturity(position_data):
    if position_data[2] >= 12:
        return False
    else:
        return True

