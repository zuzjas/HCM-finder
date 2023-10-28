import ast

import numpy as np
import pandas as pd
import wfdb
from tqdm import tqdm


def load_raw_data(data_file, sampling_rate, path):
    if sampling_rate == 100:
        data = [wfdb.rdsamp(path + f) for f in
                tqdm(data_file.filename_lr)]  # tqdm - biblioteka pokazujaca pasek progressu w konsoli
    else:
        data = [wfdb.rdsamp(path + f) for f in tqdm(data_file.filename_hr)]

    data = np.array([signal for signal, meta in data])
    return data


# Wczytaj i  przekonwertuj dane do adnotacji
# adnotowanie danych to przypisywanie danym etykiet, co pomaga modelowi w podejmowaniu decyzji i akcji
def load_and_convert_data(path, sampling_rate):
    Y = pd.read_csv(path + 'ptbxl_database.csv', index_col='ecg_id')
    # kolumna scp_codes - znajduje sie tam raport lekarza albo automatycznego interpretatora z urządzenia EKG w
    # formie stan:prawdopodobienstwo
    Y.scp_codes = Y.scp_codes.apply(lambda x: ast.literal_eval(x))  # Ocen dane w kolumnie scp_codes

    # Wczytaj surowe dane
    X = load_raw_data(Y, sampling_rate, path)

    agg_df = load_data_for_diagnostics(path)

    # agregacja danych podstawowych z diagnostycznymi
    # Agregacja klas diagnostycznych
    def aggregate_supclass_diagnostic(y_dic):
        tmp = []
        for key in y_dic.keys():
            if key in agg_df.index:
                tmp.append(agg_df.loc[key].diagnostic_class)
        return list(set(tmp))

    Y['diagnostic_superclass'] = Y.scp_codes.apply(aggregate_supclass_diagnostic)
    Y['diagnostic_superclass_len'] = Y['diagnostic_superclass'].apply(len)

    def aggregate_subclass_diagnostic(y_dic):
        tmp = []
        for key in y_dic.keys():
            if key in agg_df.index:
                tmp.append(agg_df.loc[key].diagnostic_subclass)
        ret = list(set(tmp))
        ret = ['sub_' + r for r in ret]  # to distinguish between subclass and superclass columns
        return ret

    # Apply diagnostic subclass
    Y['diagnostic_subclass'] = Y.scp_codes.apply(aggregate_subclass_diagnostic)
    Y['diagnostic_subclass_len'] = Y['diagnostic_subclass'].apply(len)

    # print(Y.loc[Y.diagnostic_superclass_len>1, 'diagnostic_superclass'])
    #plot_single_ECG(X, Y)

    return X, Y


# Funkcja pomocnicza wyswietlajaca pojedynczy sygnal EKG
def plot_single_ECG(X, Y):
    print(X.shape, Y.shape)
    print("This means that each ecg_id in Y has corresponding indetical signal in X")
    ecg_record = 1239
    Lead_II_data = X[ecg_record][1]  # Index of Lead. It is available in meta_data information
    ecg_info = Y.iloc[ecg_record]
    pd.DataFrame(Lead_II_data).plot()
    print(ecg_info)


# Wczytaj plik scp_statments.csv do agresgacji danych diagnostycnych
def load_data_for_diagnostics(path):
    agg_df = pd.read_csv(path + 'scp_statements.csv', index_col=0)
    agg_df = agg_df[agg_df.diagnostic == 1]
    return agg_df


# Modyfikacja danych do eksploracji danych
# EDA - Explorary Data Analysis
def reformat_data_for_EDA(path, sampling_rate):
    X, Y = load_and_convert_data(path, sampling_rate)

    all_superclass = pd.Series(np.concatenate(Y['diagnostic_superclass'].values))
    all_subclass = pd.Series(np.concatenate(Y['diagnostic_subclass'].values))
    superclass_cols = all_superclass.unique()
    subclass_cols = all_subclass.unique()
    update_cols = np.concatenate([superclass_cols, subclass_cols])  # dodaj kolumny metadanych
    meta_cols = ['age', 'sex', 'height', 'weight', 'nurse', 'site', 'device', ]  # moze dodac wiecej kolumn jako cechy
    x_all, y_all = get_data_by_folds(np.arange(1, 11), X, Y, update_cols, meta_cols)

    print(y_all)


# Klasa pomocnicza
class ClassUpdate:
    def __init__(self, cols):
        self.cols = cols

    def __call__(self, row):
        for sc in row['diagnostic_superclass']:
            row[sc] = 1
        for sc in row['diagnostic_subclass']:
            row[sc] = 1

        return row

def get_data_by_folds(folds, x, y, update_cols, feature_cols):
    assert len(folds) > 0, '# of provided folds should longer than 1'
    filt = np.isin(y.strat_fold.values, folds)
    x_selected = x[filt]
    y_selected = y[filt]

    for sc in update_cols:
        y_selected[sc] = 0

    cls_updt = ClassUpdate(update_cols)

    y_selected = y_selected.apply(cls_updt, axis=1)

    return x_selected, y_selected[list(feature_cols) + list(update_cols) + ['strat_fold']]
