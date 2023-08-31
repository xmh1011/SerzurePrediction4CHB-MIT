# https://medium.com/@naomi.fridman/install-conda-tensorflow-gpu-and-keras-on-ubuntu-18-04-1b403e740e25 install gpu

# start code to train the network on the CPU
import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"] = ""
# end of code to train the network on the CPU

import keras
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Conv3D, Dropout, Flatten, BatchNormalization
from keras.callbacks import EarlyStopping
from random import shuffle
import math

# to plot the model
# from keras.utils.vis_utils import plot_model
# from keras.models import load_model
# Returns a compiled model identical to the saved one
# model = load_model('my_model.h5')

PathSpectogramFolder = ''
OutputPath = ''
OutputPathModels = ''
interictalSpectograms = []
preictalSpectograms = []  # This array contains synthetic data, it is created to have a balance dataset, and it's used for training.
preictalRealSpectograms = []  # This array contains the real preictal data, it is used for testing.
patients = ["01", "02", "05", "19", "21", "23"]
nSeizure = 0


def loadParametersFromFile(filePath):
    global PathSpectogramFolder
    global OutputPath
    global OutputPathModels
    if os.path.isfile(filePath):
        with open(filePath) as f:
            line = f.readline()
            if line.split(":")[0] == "PathSpectogramFolder":
                PathSpectogramFolder = line.split(":")[1].strip()
            line = f.readline()
            if line.split(":")[0] == "OutputPath":
                OutputPath = line.split(":")[1].strip()
            line = f.readline()
            if line.split(":")[0] == "OutputPathModels":
                OutputPathModels = line.split(":")[1].strip()


def loadSpectogramData(indexPat):
    global interictalSpectograms
    global preictalSpectograms
    global preictalRealSpectograms
    global nSeizure
    nFileForSeizure = 0

    interictalSpectograms = []
    preictalSpectograms = []
    preictalRealSpectograms = []

    f = open(PathSpectogramFolder + '/paz' + patients[indexPat] + '/legendAllData.txt')
    line = f.readline()
    while "SEIZURE" not in line:
        line = f.readline()
    nSeizure = int(line.split(":")[1].strip())
    line = f.readline()
    line = f.readline()  # Read the number of spectograms.
    nSpectograms = int(line.strip())
    nFileForSeizure = math.ceil(math.ceil(nSpectograms / 50) / nSeizure)
    line = f.readline()  # Read the path of the first file

    # Reading interictal path files
    cont = -1
    indFilePathRead = 0
    while "npy" in line and indFilePathRead < nSeizure * nFileForSeizure:
        if indFilePathRead % nFileForSeizure == 0:
            interictalSpectograms.append([])
            cont = cont + 1
            interictalSpectograms[cont].append(line.split(' ')[2].rstrip())  # .rstrip() remove \n
            indFilePathRead = indFilePathRead + 1
        else:
            if len(line.split(' ')) >= 3:
                interictalSpectograms[cont].append(line.split(' ')[2].rstrip())
            indFilePathRead = indFilePathRead + 1

        line = f.readline()
    line = f.readline()  # leggo PREICTAL
    line = f.readline()  # leggo n° spectogram
    line = f.readline()  # leggo n°seizure(SEIZURE X)

    # Read path files preictal
    cont = -1
    indFilePathRead = 0
    # while(line and indFilePathRead<nSeizure*nFileForSeizure):
    while line.strip() != "":
        if "SEIZURE" in line:
            line = f.readline()  # ho letto n°seizure(SEIZURE X) perciò scorro in avanti
            if len(line.split(' ')) >= 3:
                preictalSpectograms.append([])
                cont = cont + 1
                preictalSpectograms[cont].append(line.split(' ')[2].rstrip())
                indFilePathRead = indFilePathRead + 1
        else:
            if len(line.split(' ')) >= 3:
                preictalSpectograms[cont].append(line.split(' ')[2].rstrip())
            indFilePathRead = indFilePathRead + 1

        line = f.readline()

    line = f.readline()  # leggo REAL_PREICTAL
    line = f.readline()  # leggo n° spectogram
    line = f.readline()  # leggo n°seizure(SEIZURE X)

    # Lettura path files Real Preictal
    cont = -1
    while line:
        if "SEIZURE" in line:
            line = f.readline()  # ho letto n°seizure(SEIZURE X) perciò scorro in avanti
            preictalRealSpectograms.append([])
            cont = cont + 1
            preictalRealSpectograms[cont].append(line.split(' ')[2].rstrip())
        else:
            preictalRealSpectograms[cont].append(line.split(' ')[2].rstrip())

        line = f.readline()
    f.close()


def createModel():
    input_shape = (1, 22, 59, 114)
    model = Sequential()
    # C1
    model.add(
        Conv3D(16, (22, 5, 5), strides=(1, 2, 2), activation='relu', data_format="channels_first",
               input_shape=input_shape))
    model.add(keras.layers.MaxPooling3D(pool_size=(1, 2, 2), data_format="channels_first", padding='same'))
    model.add(BatchNormalization())

    # C2
    model.add(Conv3D(32, (1, 3, 3), strides=(1, 1, 1), data_format="channels_first",
                     activation='relu'))  # incertezza se togliere padding
    model.add(keras.layers.MaxPooling3D(pool_size=(1, 2, 2), data_format="channels_first", ))
    model.add(BatchNormalization())

    # C3
    model.add(Conv3D(64, (1, 3, 3), strides=(1, 1, 1), data_format="channels_first",
                     activation='relu'))  # incertezza se togliere padding
    model.add(keras.layers.MaxPooling3D(pool_size=(1, 2, 2), data_format="channels_first", ))
    model.add(BatchNormalization())

    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(256, activation='sigmoid'))
    model.add(Dropout(0.5))
    model.add(Dense(2, activation='softmax'))

    opt_adam = tf.keras.optimizers.legacy.Adam(learning_rate=0.00001, beta_1=0.9, beta_2=0.999, epsilon=1e-08,
                                               decay=0.0)
    model.compile(loss='categorical_crossentropy', optimizer=opt_adam, metrics=['acc'])

    return model


def getFilesPathWithoutSeizure(indexSeizure):
    filesPath = []
    for i in range(0, nSeizure):  # nSeizure = 7
        if i != indexSeizure:
            if i < len(interictalSpectograms):
                filesPath.extend(interictalSpectograms[i])  # interictalSpectograms = 2
            filesPath.extend(preictalSpectograms[i])  # preictalSpectograms = 7
    shuffle(filesPath)
    return filesPath


def generate_arrays_for_training(paths, start=0, end=100):
    while True:
        from_ = int(len(paths) / 100 * start)
        to_ = int(len(paths) / 100 * end)
        for i in range(from_, int(to_)):
            f = paths[i]
            x = np.load(PathSpectogramFolder + f)
            x = np.array([x])
            x = x.swapaxes(0, 1)
            if 'P' in f:
                y = np.repeat([[0, 1]], x.shape[0], axis=0)
            else:
                y = np.repeat([[1, 0]], x.shape[0], axis=0)
            yield x, y


def generate_arrays_for_predict(paths, start=0, end=100):
    while True:
        from_ = int(len(paths) / 100 * start)
        to_ = int(len(paths) / 100 * end)
        for i in range(from_, int(to_)):
            f = paths[i]
            x = np.load(PathSpectogramFolder + f)
            x = np.array([x])
            x = x.swapaxes(0, 1)
            yield x


class EarlyStoppingByLossVal(keras.callbacks.Callback):
    def __init__(self, monitor='val_loss', value=0.00001, verbose=0, lower=True):
        super(keras.callbacks.Callback, self).__init__()
        super().__init__()
        self.monitor = monitor
        self.value = value
        self.verbose = verbose
        self.lower = lower

    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            logs = {}
        current = logs.get(self.monitor)
        if self.lower:
            if current < self.value:
                if self.verbose > 0:
                    print("Epoch %05d: early stopping THR" % epoch)
                self.model.stop_training = True
        else:
            if current > self.value:
                if self.verbose > 0:
                    print("Epoch %05d: early stopping THR" % epoch)
                self.model.stop_training = True


def main():
    print("START")
    loadParametersFromFile("PARAMETERS_CNN.txt")
    if not os.path.exists(OutputPathModels):
        os.makedirs(OutputPathModels)
    # callback=EarlyStopping(monitor='val_acc', min_delta=0, patience=0, verbose=0, mode='auto', baseline=None)
    callback = EarlyStoppingByLossVal(monitor='val_acc', value=0.975, verbose=1, lower=False)
    print("Parameters loaded")

    for indexPat in range(0, len(patients)):
        print('Patient ' + patients[indexPat])
        if not os.path.exists(OutputPathModels + "ModelPat" + patients[indexPat] + "/"):
            os.makedirs(OutputPathModels + "ModelPat" + patients[indexPat] + "/")
        loadSpectogramData(indexPat)
        print('Spectograms data loaded')

        result = 'Patient ' + patients[indexPat] + '\n'
        result = 'Out Seizure, True Positive, False Positive, False negative, Second of Inter in Test, Sensitivity, FPR \n'
        for i in range(0, nSeizure):
            print('SEIZURE OUT: ' + str(i + 1))

            print('Training start')
            model = createModel()
            filesPath = getFilesPathWithoutSeizure(i)

            model.fit_generator(generate_arrays_for_training(filesPath, end=75),
                                validation_data=generate_arrays_for_training(filesPath, start=75),
                                steps_per_epoch=int((len(filesPath) - int(len(filesPath) / 100 * 25))),
                                validation_steps=int((len(filesPath) - int(len(filesPath) / 100 * 75))), verbose=2,
                                epochs=300, max_queue_size=2, callbacks=[
                    callback])  # 100 epochs are better. Add stop criteria based on accuracy.
            print('Training end')

            print('Testing start')
            if i < len(interictalSpectograms):
                filesPath = interictalSpectograms[i]
            else:
                filesPath = []
            interictalPrediction = model.predict_generator(generate_arrays_for_predict(filesPath),
                                                           max_queue_size=4, steps=len(filesPath))
            filesPath = preictalRealSpectograms[i]
            preictalPrediction = model.predict_generator(generate_arrays_for_predict(filesPath),
                                                         max_queue_size=4, steps=len(filesPath))
            print('Testing end')

            # Creates a HDF5 file
            model.save(
                OutputPathModels + "ModelPat" + patients[indexPat] + "/" + 'ModelOutSeizure' + str(i + 1) + '.h5')
            print("Model saved")

            # to plot the model
            # plot_model(model, to_file="CNNModel", show_shapes=True, show_layer_names=True)

            if not os.path.exists(OutputPathModels + "OutputTest" + "/"):
                os.makedirs(OutputPathModels + "OutputTest" + "/")
            np.savetxt(OutputPathModels + "OutputTest" + "/" + "Int_" + patients[indexPat] + "_" + str(i + 1) + ".csv",
                       interictalPrediction, delimiter=",")
            np.savetxt(OutputPathModels + "OutputTest" + "/" + "Pre_" + patients[indexPat] + "_" + str(i + 1) + ".csv",
                       preictalPrediction, delimiter=",")

            secondsInterictalInTest = len(
                interictalSpectograms[i]) * 50 * 30  # 50 spectograms for file, 30 seconds for each spectogram
            acc = 0  # accumulator
            fp = 0
            tp = 0
            fn = 0
            lastTenResult = list()

            for el in interictalPrediction:
                if el[1] > 0.5:
                    acc = acc + 1
                    lastTenResult.append(1)
                else:
                    lastTenResult.append(0)
                if len(lastTenResult) > 10:
                    acc = acc - lastTenResult.pop(0)
                if acc >= 8:
                    fp = fp + 1
                    lastTenResult = list()
                    acc = 0

            lastTenResult = list()
            for el in preictalPrediction:
                if el[1] > 0.5:
                    acc = acc + 1
                    lastTenResult.append(1)
                else:
                    lastTenResult.append(0)
                if len(lastTenResult) > 10:
                    acc = acc - lastTenResult.pop(0)
                if acc >= 8:
                    tp = tp + 1
                else:
                    if len(lastTenResult) == 10:
                        fn = fn + 1

            sensitivity = tp / (tp + fn)
            FPR = fp / (secondsInterictalInTest / (60 * 60))

            result = result + str(i + 1) + ',' + str(tp) + ',' + str(fp) + ',' + str(fn) + ',' + str(
                secondsInterictalInTest) + ','
            result = result + str(sensitivity) + ',' + str(FPR) + '\n'
            print('True Positive, False Positive, False negative, Second of Inter in Test, Sensitivity, FPR')
            print(str(tp) + ',' + str(fp) + ',' + str(fn) + ',' + str(secondsInterictalInTest) + ',' + str(
                sensitivity) + ',' + str(FPR))
        with open(OutputPath, "a+") as myFile:
            myFile.write(result)


if __name__ == '__main__':
    main()
