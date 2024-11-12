import os
# disable keras loggings
import sys
stderr = sys.stderr
sys.stderr = open(os.devnull, 'w')
import tensorflow as tf

from tensorflow.keras.layers import LSTM, GRU, Dense, Activation, LeakyReLU, Dropout
from tensorflow.keras.layers import Conv1D, MaxPool1D, GlobalAveragePooling1D
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.keras.utils import to_categorical

from sklearn.metrics import accuracy_score, mean_absolute_error, confusion_matrix

from data_extractor import load_data
from create_csv import write_custom_csv, write_emodb_csv, write_tess_ravdess_csv
from emotion_recognition import EmotionRecognizer
from utilsspeech import get_first_letters, AVAILABLE_EMOTIONS, extract_feature, get_dropout_str

import numpy as np
import pandas as pd
import random


class DeepEmotionRecognizer(EmotionRecognizer):
    """
    نسخه یادگیری عمیق 
    استفاده از لایه هایRNN (LSTM, GRU, etc.) and Dense .
    
    """
    def __init__(self, **kwargs):


        super().__init__(**kwargs)

        self.n_rnn_layers = kwargs.get("n_rnn_layers", 2)
        self.n_dense_layers = kwargs.get("n_dense_layers", 2)
        self.rnn_units = kwargs.get("rnn_units", 128)
        self.dense_units = kwargs.get("dense_units", 128)
        self.cell = kwargs.get("cell", LSTM)

        self.dropout = kwargs.get("dropout", 0.3)
        self.dropout = self.dropout if isinstance(self.dropout, list) else [self.dropout] * ( self.n_rnn_layers + self.n_dense_layers )

        self.output_dim = len(self.emotions)

        # ویژگی های بهینه سازی
        self.optimizer = kwargs.get("optimizer", "adam")
        self.loss = kwargs.get("loss", "categorical_crossentropy")

        # ویزگی های آموزشی 
        self.batch_size = kwargs.get("batch_size", 64)
        self.epochs = kwargs.get("epochs", 500)
        

        self.model_name = ""
        self._update_model_name()

        self.model = None


        self._compute_input_length()


        self.model_created = False

    def _update_model_name(self):
        """
یک نام منحصر به فرد بر اساس پارامتر های داده شده برای مدل ایجاد می کند
        """
        # اولین حروف احساسات را برمیگرداند
        # ["sad", "neutral", "happy"] => 'HNS' (sorted alphabetically)
        emotions_str = get_first_letters(self.emotions)
        # 'c' برای طبقه بندی & 'r' برای رگرسیون
        problem_type = 'c' if self.classification else 'r'
        dropout_str = get_dropout_str(self.dropout, n_layers=self.n_dense_layers + self.n_rnn_layers)
        self.model_name = f"{emotions_str}-{problem_type}-{self.cell.__name__}-layers-{self.n_rnn_layers}-{self.n_dense_layers}-units-{self.rnn_units}-{self.dense_units}-dropout-{dropout_str}.h5"

    def _get_model_filename(self):
        """مسیر مدل را مشخص میکند"""
        return f"results/{self.model_name}"

    def _model_exists(self):
        """
        بررسی میکند مدل قبلا وجود  داشته است
        """
        filename = self._get_model_filename()
        return filename if os.path.isfile(filename) else None

    def _compute_input_length(self):
        """
        شکل ورودی را محسابه میکند تا مدل را بسازد
        """
        if not self.data_loaded:
            self.load_data()
        self.input_length = self.X_train[0].shape[1]

    def _verify_emotions(self):
        super()._verify_emotions()
        self.int2emotions = {i: e for i, e in enumerate(self.emotions)}
        self.emotions2int = {v: k for k, v in self.int2emotions.items()}

    def create_model(self):
        """
        شبکه عصبی را بر اساس پارامتر ها می سازد
        """
        if self.model_created:
            # مدل قبلا بارگیری شده است
            return

        if not self.data_loaded:
            # اگر اطلاعات بارگیری نشده اند بارگیری میکند
            self.load_data()
        
        model = Sequential()

        # لایه ها
        for i in range(self.n_rnn_layers):
            if i == 0:
                # لایه اول
                model.add(self.cell(self.rnn_units, return_sequences=True, input_shape=(None, self.input_length)))
                model.add(Dropout(self.dropout[i]))
            else:
                # لایه های میانی
                model.add(self.cell(self.rnn_units, return_sequences=True))
                model.add(Dropout(self.dropout[i]))

        if self.n_rnn_layers == 0:
            i = 0

        # لایه های متراکم
        for j in range(self.n_dense_layers):
            
            if self.n_rnn_layers == 0 and j == 0:
                model.add(Dense(self.dense_units, input_shape=(None, self.input_length)))
                model.add(Dropout(self.dropout[i+j]))
            else:
                model.add(Dense(self.dense_units))
                model.add(Dropout(self.dropout[i+j]))
                
        if self.classification:
            model.add(Dense(self.output_dim, activation="softmax"))
            model.compile(loss=self.loss, metrics=["accuracy"], optimizer=self.optimizer)
        else:
            model.add(Dense(1, activation="linear"))
            model.compile(loss="mean_squared_error", metrics=["mean_absolute_error"], optimizer=self.optimizer)
        
        self.model = model
        self.model_created = True
        if self.verbose > 0:
            print("[+] Model created")

    def load_data(self):
        """
        ویژگی های استخراج شده را بارگیری میگند
        """
        super().load_data()
        X_train_shape = self.X_train.shape
        X_test_shape = self.X_test.shape
        self.X_train = self.X_train.reshape((1, X_train_shape[0], X_train_shape[1]))
        self.X_test = self.X_test.reshape((1, X_test_shape[0], X_test_shape[1]))

        if self.classification:

            self.y_train = to_categorical([ self.emotions2int[str(e)] for e in self.y_train ])
            self.y_test = to_categorical([ self.emotions2int[str(e)] for e in self.y_test ])
        
        # تغییر شکل برچسب ها
        y_train_shape = self.y_train.shape
        y_test_shape = self.y_test.shape
        if self.classification:
            self.y_train = self.y_train.reshape((1, y_train_shape[0], y_train_shape[1]))    
            self.y_test = self.y_test.reshape((1, y_test_shape[0], y_test_shape[1]))
        else:
            self.y_train = self.y_train.reshape((1, y_train_shape[0], 1))
            self.y_test = self.y_test.reshape((1, y_test_shape[0], 1))

    def train(self, override=False):

        # if model isn't created yet, create it
        if not self.model_created:
            self.create_model()

        # اگر مدل آموزش دیده موجود است فقط وزن ها را بارگیری میکند
        # درغیر این صورت از بارگیری وزن ها صرف نظر می کند
        if not override:
            model_name = self._model_exists()
            if model_name:
                self.model.load_weights(model_name)
                self.model_trained = True
                if self.verbose > 0:
                    print("[*] Model weights loaded")
                return
        
        if not os.path.isdir("results"):
            os.mkdir("results")

        if not os.path.isdir("logs"):
            os.mkdir("logs")

        model_filename = self._get_model_filename()

        self.checkpointer = ModelCheckpoint(model_filename, save_best_only=True, verbose=1)
        self.tensorboard = TensorBoard(log_dir=os.path.join("logs", self.model_name))

        self.history = self.model.fit(self.X_train, self.y_train,
                        batch_size=self.batch_size,
                        epochs=self.epochs,
                        validation_data=(self.X_test, self.y_test),
                        callbacks=[self.checkpointer, self.tensorboard],
                        verbose=self.verbose)
        
        self.model_trained = True
        if self.verbose > 0:
            print("[+] Model trained")

    def predict(self, audio_path):
        feature = extract_feature(audio_path, **self.audio_config).reshape((1, 1, self.input_length))
        if self.classification:
            prediction = self.model.predict(feature)
            prediction = np.argmax(np.squeeze(prediction))
            return self.int2emotions[prediction]
        else:
            return np.squeeze(self.model.predict(feature))

    def predict_proba(self, audio_path):
        if self.classification:
            feature = extract_feature(audio_path, **self.audio_config).reshape((1, 1, self.input_length))
            proba = self.model.predict(feature)[0][0]
            result = {}
            for prob, emotion in zip(proba, self.emotions):
                result[emotion] = prob
            return result
        else:
            raise NotImplementedError("Probability prediction doesn't make sense for regression")



    def test_score(self):
        y_test = self.y_test[0]
        if self.classification:
            y_pred = self.model.predict(self.X_test)[0]
            y_pred = [np.argmax(y, out=None, axis=None) for y in y_pred]
            y_test = [np.argmax(y, out=None, axis=None) for y in y_test]
            return accuracy_score(y_true=y_test, y_pred=y_pred)
        else:
            y_pred = self.model.predict(self.X_test)[0]
            return mean_absolute_error(y_true=y_test, y_pred=y_pred)

    def train_score(self):
        y_train = self.y_train[0]
        if self.classification:
            y_pred = self.model.predict(self.X_train)[0]
            y_pred = [np.argmax(y, out=None, axis=None) for y in y_pred]
            y_train = [np.argmax(y, out=None, axis=None) for y in y_train]
            return accuracy_score(y_true=y_train, y_pred=y_pred)
        else:
            y_pred = self.model.predict(self.X_train)[0]
            return mean_absolute_error(y_true=y_train, y_pred=y_pred)

    def confusion_matrix(self, percentage=True, labeled=True):
        """محاسبه ماتریس درهم ریختگی برای دقت آزمون """
        if not self.classification:
            raise NotImplementedError("Confusion matrix works only when it is a classification problem")
        y_pred = self.model.predict(self.X_test)[0]
        y_pred = np.array([ np.argmax(y, axis=None, out=None) for y in y_pred])
        y_test = np.array([ np.argmax(y, axis=None, out=None) for y in self.y_test[0] ])
        matrix = confusion_matrix(y_test, y_pred, labels=[self.emotions2int[e] for e in self.emotions]).astype(np.float32)
        if percentage:
            for i in range(len(matrix)):
                matrix[i] = matrix[i] / np.sum(matrix[i])
            # تبدیل به درصد
            matrix *= 100
        if labeled:
            matrix = pd.DataFrame(matrix, index=[ f"true_{e}" for e in self.emotions ],
                                    columns=[ f"predicted_{e}" for e in self.emotions ])
        return matrix

    def get_n_samples(self, emotion, partition):
        """برگرداندن عدد احساسات نمونه ها
        """
        if partition == "test":
            if self.classification:
                y_test = np.array([ np.argmax(y, axis=None, out=None)+1 for y in np.squeeze(self.y_test) ]) 
            else:
                y_test = np.squeeze(self.y_test)
            return len([y for y in y_test if y == emotion])
        elif partition == "train":
            if self.classification:
                y_train = np.array([ np.argmax(y, axis=None, out=None)+1 for y in np.squeeze(self.y_train) ])
            else:
                y_train = np.squeeze(self.y_train)
            return len([y for y in y_train if y == emotion])

    def get_samples_by_class(self):
        """
        تعداد آموزش های نمونه ها را برمیگرداند
        """
        train_samples = []
        test_samples = []
        total = []
        for emotion in self.emotions:
            n_train = self.get_n_samples(self.emotions2int[emotion]+1, "train")
            n_test = self.get_n_samples(self.emotions2int[emotion]+1, "test")
            train_samples.append(n_train)
            test_samples.append(n_test)
            total.append(n_train + n_test)
        
        # بدست آوردن کل نمونه ها
        total.append(sum(train_samples) + sum(test_samples))
        train_samples.append(sum(train_samples))
        test_samples.append(sum(test_samples))
        return pd.DataFrame(data={"train": train_samples, "test": test_samples, "total": total}, index=self.emotions + ["total"])

    def get_random_emotion(self, emotion, partition="train"):
        
        if partition == "train":
            y_train = self.y_train[0]
            index = random.choice(list(range(len(y_train))))
            element = self.int2emotions[np.argmax(y_train[index])]
            while element != emotion:
                index = random.choice(list(range(len(y_train))))
                element = self.int2emotions[np.argmax(y_train[index])]
        elif partition == "test":
            y_test = self.y_test[0]
            index = random.choice(list(range(len(y_test))))
            element = self.int2emotions[np.argmax(y_test[index])]
            while element != emotion:
                index = random.choice(list(range(len(y_test))))
                element = self.int2emotions[np.argmax(y_test[index])]
        else:
            raise TypeError("Unknown partition, only 'train' or 'test' is accepted")

        return index

    def determine_best_model(self):
        pass


if __name__ == "__main__":
    rec = DeepEmotionRecognizer(emotions=['angry', 'sad', 'neutral', 'ps', 'happy'],
                                epochs=300, verbose=0)
    rec.train(override=False)
    print("Test accuracy score:", rec.test_score() * 100, "%")