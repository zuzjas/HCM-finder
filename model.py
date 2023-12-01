import ast
import sys

import numpy as np
import pandas as pd
import wfdb
from keras import Sequential
from keras.src import callbacks, losses
from keras.src.layers import Dense, Conv1D, BatchNormalization, LeakyReLU, MaxPool1D, Dropout, AveragePooling1D, GlobalAveragePooling1D
from keras.src.saving.legacy.saved_model.save_impl import metrics
from matplotlib import pyplot as plt
from sklearn.base import TransformerMixin
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import layers, optimizers, losses, metrics, regularizers, callbacks
from keras.models import Model
from tqdm import tqdm
import seaborn as sns


def load_raw_data(data_file, sampling_rate, path):
    if sampling_rate == 100:
        data = [wfdb.rdsamp(path + f) for f in tqdm(data_file.filename_lr)]  # tqdm - biblioteka pokazujaca pasek progressu w konsoli
    else:
        data = [wfdb.rdsamp(path + f) for f in tqdm(data_file.filename_hr)]

    data = np.array([signal for signal, meta in data])
    return data


# Funkcja pokazuje pojedyncza probke sygnalu EKG
def show_one_sample_from_ECG(ECG_data, sample_number):
    sample = ECG_data[sample_number]
    bar, axes = plt.subplots(sample.shape[1], 1, figsize=(20, 10))
    for i in range(sample.shape[1]):
        sns.lineplot(x=np.arange(sample.shape[0]), y=sample[:, i], ax=axes[i])

    plt.show()


def load_data(ECG_df, SCP_df, PATH_TO_DATA):
    # Wczytanie metadanych
    ECG_df.scp_codes = ECG_df.scp_codes.apply(lambda x: ast.literal_eval(x))
    ECG_df.patient_id = ECG_df.patient_id.astype(int)
    ECG_df.nurse = ECG_df.nurse.astype('Int64')
    ECG_df.site = ECG_df.site.astype('Int64')
    ECG_df.validated_by = ECG_df.validated_by.astype('Int64')

    SCP_df = SCP_df[SCP_df.diagnostic == 1]

    # Dodanie dodatkowej kolumny do ECG_df z klasami chorobowymi sygnalow
    def diagnostic_class(scp):
        res = set()
        for k in scp.keys():
            if k in SCP_df.index:
                res.add(SCP_df.loc[k].diagnostic_class)
        return list(res)

    ECG_df['scp_classes'] = ECG_df.scp_codes.apply(diagnostic_class)

    # Zaladowanie danych sygnalu EKG do ECG_data
    sampling_rate = 100
    ECG_data = load_raw_data(ECG_df, sampling_rate, PATH_TO_DATA)

    return ECG_df, ECG_data


# Dzieli wczytane dane na podzbiory
def prepare_data(ECG_df, ECG_data):
    # Podzial na zbiory: X,Y
    # X - sygnaly EKG
    # Y - diagnoza: 1-HCM obecne; 0-HCM nieobecne

    # X - zostaje bez zmian; bedzie to po prostu zbior ECG_data
    X = ECG_data

    # Y
    Y = pd.DataFrame(0, index=ECG_df.index, columns=['HYP'], dtype='int')
    for i in Y.index:
        for j in ECG_df.loc[i].scp_classes:
            if j == 'HYP':
                Y.loc[i, j] = 1

    # Podzial na zbiory train, vaidate, test
    # podzial zgodnie z kolumna strat_fold
    X_train, Y_train = ECG_data[ECG_df.strat_fold <= 8], Y[ECG_df.strat_fold <= 8]
    X_valid, Y_valid = ECG_data[ECG_df.strat_fold == 9], Y[ECG_df.strat_fold == 9]
    X_test, Y_test = ECG_data[ECG_df.strat_fold == 10], Y[ECG_df.strat_fold == 10]

    # Zapis przygotowanych danych do plikow .npz
    NUMPY_DATA_FILE = 'output/data.npz'

    save_args = {
        'X_train': X_train.astype('float32'),
        'X_valid': X_valid.astype('float32'),
        'X_test': X_test.astype('float32'),
        'Y_train': Y_train.to_numpy().astype('float32'),
        'Y_valid': Y_valid.to_numpy().astype('float32'),
        'Y_test': Y_test.to_numpy().astype('float32'),
    }
    np.savez(NUMPY_DATA_FILE, **save_args)


def read_data_from_npz(npz_path):
    thismodule = sys.modules[__name__]

    with np.load(npz_path) as data:
        for k in data.keys():
            setattr(thismodule, k, data[k].astype(float))


# Prostszy model
def cnnmodel():
    model = Sequential()

    model.add(Conv1D(filters=5, kernel_size=3, strides=1, input_shape=(1000, 12)))
    model.add(BatchNormalization())
    model.add(LeakyReLU())
    model.add(MaxPool1D(pool_size=2, strides=2))

    model.add(Conv1D(filters=5, kernel_size=3, strides=1))
    model.add(LeakyReLU())
    model.add(MaxPool1D(pool_size=2, strides=2))
    model.add(Dropout(0.5))

    model.add(Conv1D(filters=5, kernel_size=3, strides=1))
    model.add(LeakyReLU())
    model.add(AveragePooling1D(pool_size=2, strides=2))
    model.add(Dropout(0.5))

    model.add(Conv1D(filters=5, kernel_size=3, strides=1))
    model.add(LeakyReLU())
    model.add(AveragePooling1D(pool_size=2, strides=2))

    model.add(Conv1D(filters=5, kernel_size=3, strides=1))
    model.add(LeakyReLU())
    model.add(GlobalAveragePooling1D())
    model.add(Dense(1, activation='sigmoid'))

    print(model.summary())
    return model


# bardziej zlozony model
def cnnmodel2():
    input = layers.Input(shape=(1000, 12))

    X = layers.Conv1D(filters=32, kernel_size=5)(input)
    X = layers.BatchNormalization()(X)
    X = layers.ReLU()(X)
    X = layers.MaxPooling1D(pool_size=2, strides=1)(X)

    convC1 = layers.Conv1D(filters=64, kernel_size=7)(X)

    X = layers.Conv1D(filters=32, kernel_size=5)(X)
    X = layers.BatchNormalization()(X)
    X = layers.ReLU()(X)
    X = layers.MaxPooling1D(pool_size=4, strides=1)(X)

    convC2 = layers.Conv1D(filters=64, kernel_size=6)(convC1)

    X = layers.Conv1D(filters=64, kernel_size=5)(X)
    X = layers.BatchNormalization()(X)
    X = layers.Add()([convC2, X])
    X = layers.ReLU()(X)
    X = layers.MaxPooling1D(pool_size=2, strides=1)(X)

    convE1 = layers.Conv1D(filters=32, kernel_size=4)(X)

    X = layers.Conv1D(filters=64, kernel_size=3)(X)
    X = layers.BatchNormalization()(X)
    X = layers.ReLU()(X)
    X = layers.MaxPooling1D(pool_size=4, strides=1)(X)

    convE2 = layers.Conv1D(filters=64, kernel_size=5)(convE1)

    X = layers.Conv1D(filters=64, kernel_size=3)(X)
    X = layers.BatchNormalization()(X)
    X = layers.Add()([convE2, X])
    X = layers.ReLU()(X)
    X = layers.MaxPooling1D(pool_size=2, strides=1)(X)

    X = layers.Conv1D(filters=64, kernel_size=1)(X)
    X = layers.BatchNormalization()(X)
    X = layers.ReLU()(X)
    X = layers.GlobalAveragePooling1D()(X)

    X = layers.Flatten()(X)

    X = layers.Dense(units=128, kernel_regularizer=regularizers.L2(0.005))(X)
    X = layers.BatchNormalization()(X)
    X = layers.ReLU()(X)
    X = layers.Dropout(rate=0.1)(X)

    X = layers.Dense(units=64, kernel_regularizer=regularizers.L2(0.009))(X)
    X = layers.BatchNormalization()(X)
    X = layers.ReLU()(X)
    X = layers.Dropout(rate=0.15)(X)

    output = layers.Dense(1, activation='sigmoid')(X)
    model = Model(inputs=input, outputs=output)
    print(model.summary())
    return model


# Funkcja do skalowania 3-wymiarowych danych
class NDStandardScaler(TransformerMixin):
    def __init__(self, **kwargs):
        self._scaler = StandardScaler(copy=True, **kwargs)
        self._orig_shape = None

    def fit(self, X, **kwargs):
        X = np.array(X)
        if len(X.shape) > 1:
            self._orig_shape = X.shape[1:]
        X = self._flatten(X)
        self._scaler.fit(X, **kwargs)
        return self

    def transform(self, X, **kwargs):
        X = np.array(X)
        X = self._flatten(X)
        X = self._scaler.transform(X, **kwargs)
        X = self._reshape(X)
        return X

    def _flatten(self, X):
        if len(X.shape) > 2:
            n_dims = np.prod(self._orig_shape)
            X = X.reshape(-1, n_dims)
        return X

    def _reshape(self, X):
        if len(X.shape) >= 2:
            X = X.reshape(-1, *self._orig_shape)
        return X


def mmain():
    # wczytanie danych i zapisanie ich do pliku .npz
    ####################################################################
    # PATH_TO_DATA = 'C:/Users/user\/Documents/Python/HCM_finder/input/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/'
    # ECG_df = pd.read_csv(os.path.join(PATH_TO_DATA, 'ptbxl_database.csv'), index_col='ecg_id')
    # SCP_df = pd.read_csv(os.path.join(PATH_TO_DATA, 'scp_statements.csv'), index_col=0)
    #
    # ECG_df, ECG_data = load_data(ECG_df, SCP_df, PATH_TO_DATA)
    # prepare_data(ECG_df, ECG_data)
    ##################################################################

    # wczytanie danych z pliku .npz
    npz_path = 'C:/Users/user\/Documents/Python/HCM_finder/output/data.npz'
    read_data_from_npz(npz_path)

    # skalowanie danych
    scaler = NDStandardScaler()
    scaler.fit(X_train)
    x_train = X_train
    x_train = scaler.transform(X_train)

    x_test = X_test
    x_test = scaler.transform(X_test)

    x_valid = X_valid
    x_valid = scaler.transform(X_valid)

    # parametry wczesniejszego zatrzymania
    early = callbacks.EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True)
    reducelr = callbacks.ReduceLROnPlateau(monitor="val_loss", patience=3)
    callback = [early, reducelr]

    # trening danych
    model = cnnmodel2()
    model.compile(optimizer=optimizers.Adam(learning_rate=0.0005), loss=losses.BinaryCrossentropy(),
                  metrics=[metrics.BinaryAccuracy(), metrics.AUC(curve='ROC', multi_label=True)])

    history = model.fit(x=x_train, y=Y_train, epochs=100, batch_size=32, callbacks=callback,
                        validation_data=(x_valid, Y_valid))

    # wykres funkcji utraty (model loss)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    accuracy = []
    accuracy.append(model.evaluate(x_valid, Y_valid)[1])

    print(' valid accuracy:', np.mean(accuracy))

    train_accuracy = model.evaluate(X_train, Y_train)
    test_accuracy = model.evaluate(X_test, Y_test)

    # test modelu
    predictions = model.predict(x=x_test)

    rounded_pressictions = predictions > 0.5

    cm = confusion_matrix(y_true=Y_test, y_pred=rounded_pressictions)
    cm_plot_labels = ['no hypertrophy', 'hypertrophy']
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=cm_plot_labels)
    disp.plot()
    plt.show()

    print("Trainining accuracy:", train_accuracy)
    print("Testing accuracy", test_accuracy)
