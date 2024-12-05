import pandas as pd
import numpy as np
import os
from keras.callbacks import EarlyStopping, ModelCheckpoint
import src.preprocess as pp
from src.my_keras_models import get_model, reset_model_states
from sklearn.svm import SVR, SVC, LinearSVR
from sklearn.multioutput import MultiOutputRegressor
from sklearn.tree import DecisionTreeRegressor

class PredictorInputs:
    def __init__(self, args, X, y, in_scaler=None, out_scaler=None, batch_padding=False):
        self.in_scaler = in_scaler
        self.out_scaler = out_scaler
        self.X_df, self.y_df = X.copy(), y.copy()
        self.pad_rows = 0

        if batch_padding and ('lstm' in args.model or 'mlp' in args.model):
            self.pad_rows = args.batch_size - (X.shape[0] % args.batch_size) if args.batch_size > 1 else 0
            if self.pad_rows > 0:
                xpad = pd.DataFrame(np.zeros((self.pad_rows, X.shape[1])), columns=X.columns)
                X = pd.concat([X, xpad], ignore_index=True)
                ypad = pd.DataFrame(np.zeros((self.pad_rows, y.shape[1])), columns=y.columns)
                y = pd.concat([y, ypad], ignore_index=True)

        # # --------------------------------------------------- SHAPE INPUT DATA --------------------------------------------------- # #
        # Scalar inputs only
        self.X, self.in_scaler = self.__class__._scale_data(X.values, args.scaler, self.in_scaler)
        self.X = self.__class__._shape_continuous_inputs(args, self.X)

        # # --------------------------------------- SHAPE OUTPUT DATA --------------------------------------- # #
        # Scalar outputs only
        self.y, self.out_scaler = self.__class__._scale_data(y.values, args.scaler, self.out_scaler)
        if args.model == "svm" or args.model == 'dt':
            if args.forecast_horizon == 1:
                # Two dimensions not supported for y data.
                self.y = self.y.reshape(self.y.shape[0],)

    def _scale_input_output(self, args, X, y):
        # Scale both X and Y
        scaler_name = args.scaler
        scaled_X, self.in_scaler = self.__class__._scale_data(X.values, scaler_name, self.in_scaler)
        scaled_y, self.out_scaler = self.__class__._scale_data(y.values, scaler_name, self.out_scaler)
        return scaled_X, scaled_y
    
    @staticmethod
    def _scale_data(data, scaler_name, scaler=None):
        if scaler_name != 'none':
            if not scaler:
                scaler = pp.get_scaler(data, scaler_name)
            scaled_data = pp.apply_scaler(scaler, data)
            return scaled_data, scaler
        return data, None

    @staticmethod
    def _shape_continuous_inputs(args, X):
        if "lstm" in args.model:
            X = X.reshape(X.shape[0], args.timesteps, int(X.shape[1] / args.timesteps))
        return X
    
    def add_predictions(self, args, y_hat):
        if self.pad_rows > 0:
            self.y_hat = y_hat[:-self.pad_rows]
        else:
            self.y_hat = y_hat
        scaler_name = args.scaler
        if scaler_name != "none":
            self.y_hat = self.out_scaler.inverse_transform(self.y_hat)
        self.y_hat = pd.DataFrame(self.y_hat, index=self.y_df.index, columns=self.y_df.columns)

class Predictor:
    def __init__(self, model):
        self.model = model
    @staticmethod
    def _get_continuous_inputs_count(args, x_shape):
        if len(x_shape) > 1:
            inputs_count = x_shape[1]
        else:
            inputs_count = 1
        if (args.perfect_rle and args.max_run_length > 0) or args.rle:
            inputs_count = 2 # TODO: this assumes RLE always use one counter
        return inputs_count
    @staticmethod
    def _get_model_name_prefix(args):
        model_name = ""
        model_name += args.model
        return model_name

    def train_predict(self, args, train_data, test_data, monitor):
        X, y = train_data.X, train_data.y
        vX, vy = test_data.X, test_data.y
        saved_model_name = os.path.join(args.results_folder,
                                        args.benchmark+'_'+str(np.random.randint(10,100)))
        if args.early_stopping:
            early_stopping_monitor = EarlyStopping(monitor=monitor, patience=args.patience)
            callbacks = [early_stopping_monitor, ModelCheckpoint(filepath=saved_model_name+'.h5', monitor=monitor, save_best_only=True)]
            history = self.model.fit(X, y, validation_data=(vX, vy),
                                    batch_size=args.batch_size,
                                    epochs=args.epochs, shuffle=True,
                                    verbose=0, callbacks=callbacks)
            reset_model_states(self.model)
            self.model.load_weights(saved_model_name+'.h5')
            if early_stopping_monitor.stopped_epoch != 0:
                print("[Predictor.py] Stopped epoch: ", early_stopping_monitor.stopped_epoch)
        else:
            history = self.model.fit(X, y, validation_data=(vX, vy),
                                    batch_size=args.batch_size,
                                    epochs=args.epochs, shuffle=True, verbose=0)
        reset_model_states(self.model)
        output = self.model.predict(vX, batch_size=args.batch_size)
        return output

    def train_predict_svm(self, train_data, test_data):
        self.model.fit(train_data.X, train_data.y.astype(int))
        output = self.model.predict(test_data.X)
        return output

class Regressor(Predictor):
    def __init__(self, args, X):
        # inputs_count = super()._get_continuous_inputs_count(args, X.shape)
        loss = args.loss_function
        dual_activation = None
        class_count = out_steps = args.forecast_horizon
        if "lstm" in args.model:
            self.input_shape = (args.batch_size, X.shape[1], X.shape[2])
        else:
            self.input_shape = (X.shape[1],)
        self.model_name = super()._get_model_name_prefix(args)
        if args.model == "svm":
            if args.svm_kernel == 'linear':
                self.model = LinearSVR(
                    C=args.svm_regularization,
                    epsilon=args.svm_epsilon,
                    max_iter=args.max_iter,
                )
            else:
                self.model = SVR(
                    kernel=args.svm_kernel,
                    C=args.svm_regularization,
                    epsilon=args.svm_epsilon,
                    max_iter=args.max_iter,
                )
        elif args.model == 'dt':
            self.model = DecisionTreeRegressor(max_depth=args.tree_max_depth)
            if args.forecast_horizon > 1:
                self.model = MultiOutputRegressor(self.model, n_jobs=-1)
        else:
            self.model = get_model(self.model_name, self.input_shape, args.neurons, not args.stateless,
                                            class_count = class_count,
                                            dense_layers = args.dense_hidden_layers,
                                            loss=loss, dual_activation=dual_activation,
                                            optimizer=args.optimizer, stacked_layers=args.stacked_layers,
                                            regression_activation=args.regression_activation)
        super().__init__(self.model)

    def train_predict_svm(self, train_data, test_data):
        # Work-around SVM takes forever above O(10^5)
        if train_data.X.size > 100000:
            samples = np.random.choice(len(train_data.X), int(len(train_data.X)/4))
            print("[Predictor.py] WARNING: Train data is too large, it will be sampled in half")
            X = train_data.X[samples]
            y = train_data.y[samples]
        else:
            X = train_data.X
            y = train_data.y
        self.model.fit(X, y)
        output = self.model.predict(test_data.X)
        return output
    
    def train_predict(self, args, train_data, test_data):
        if args.model == "svm" or args.model == 'dt':
            predictions = self.train_predict_svm(train_data, test_data)
            if args.forecast_horizon == 1:
                predictions = predictions.reshape(-1,1)
        else:
            predictions = super().train_predict(args, train_data, test_data, "val_loss")
        return predictions


class SerialPredictor():
    def __init__(self, args, X):
        self.predictor = Regressor(args,X)
    def train_predict(self, args, train_data, test_data):
        return self.predictor.train_predict(args, train_data, test_data)
