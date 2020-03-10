from keras import Sequential
from keras.callbacks import EarlyStopping
from keras.layers import Masking, LSTM, Dropout, Dense
from keras.optimizers import Adam
import matplotlib.pyplot as plt

es = EarlyStopping(monitor='loss', mode='min', verbose=1, restore_best_weights=True, patience=5)

class WorkshopLSTM:
    def __init__(self, params):
        self.learning_rate = params["learning_rate"]
        self.lstm_cells = params["lstm_cells"]
        self.dropout = params["dropout"]
        self.epochs = params["epochs"]
        self.model = None
        self.history = None
        self.scores = None

    def build(self, x_train):
        opt = Adam(learning_rate=self.learning_rate)
        self.model = Sequential()
        self.model.add(Masking(mask_value=-1, input_shape=(x_train.shape[1], 1)))
        self.model.add(LSTM(self.lstm_cells, recurrent_dropout=self.dropout))
        self.model.add(Dropout(self.dropout))
        self.model.add(Dense(1, activation='sigmoid'))
        self.model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['acc'])
        print(self.model.summary())

    def fit_and_evaluate(self, x_train, y_train, x_test, y_test):
        self.history = self.model.fit(
            x=x_train,
            y=y_train,
            epochs=self.epochs,
            verbose=1,
            callbacks=[es],
            validation_data=(x_test, y_test),
        )
        self.scores = self.model.evaluate(x_test, y_test, verbose=0)

    def plot(self):
        plt.rcParams["figure.figsize"] = (16, 5)
        fig, (ax1, ax2) = plt.subplots(1, 2)

        ax1.plot(self.history.history['loss'], label='Training')
        ax1.plot(self.history.history['val_loss'], label='Testing')
        ax1.legend(loc="upper left")
        ax1.set_title("Training und Testing Kostenverlauf")
        ax2.plot(self.history.history['acc'], label='Training')
        ax2.plot(self.history.history['val_acc'], label='Testing')
        ax2.legend(loc="upper left")  #
        ax2.set_title("Training und Testing Genauigkeit")
        plt.show()

    def print_scores(self, x_test):
        print("Genauigkeit: %.2f%%" % (self.scores[1] * 100))
        print(str(round(x_test.shape[0] * self.scores[1])) + " von " + str(
            x_test.shape[0]) + " Beispielen richtig klassifiziert!")
