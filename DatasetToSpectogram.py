import pyedflib
import numpy as np
from scipy import signal
from scipy.signal import butter, lfilter
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import os

# DATASET: https://physionet.org/pn6/chbmit/
sampleRate = 256
pathDataSet = ''  # path of the dataset
FirstPartPathOutput = ''  # path where the spectogram will be saved
# patients = ["01", "02", "03", "05", "09", "10", "13", "14", "18", "19", "20", "21", "23"]
# nSeizure = [7, 3, 6, 5, 4, 6, 5, 5, 6, 3, 5, 4, 5]
patients = ["01", "02", "05", "19", "21", "23"]
_30_MINUTES_OF_DATA = 256 * 60 * 30
_MINUTES_OF_DATA_BETWEEN_PRE_AND_SEIZURE = 3  # In teoria 5 come l'SPH ma impostato a 3 per considerare alcune seizure prese nel paper
_MINUTES_OF_PREICTAL = 30
_SIZE_WINDOW_IN_SECONDS = 30
_SIZE_WINDOW_SPECTOGRAM = _SIZE_WINDOW_IN_SECONDS * 256
nSpectrogram = 0
signalsBlock = None
SecondPartPathOutput = ''
legendOfOutput = ''
isPreictal = ''


def loadParametersFromFile(filePath):
    global pathDataSet
    global FirstPartPathOutput
    if os.path.isfile(filePath):
        with open(filePath) as f:
            line = f.readline()
            if line.split(":")[0] == "pathDataSet":
                pathDataSet = line.split(":")[1].strip()
            line = f.readline()
            if line.split(":")[0] == "FirstPartPathOutput":
                FirstPartPathOutput = line.split(":")[1].strip()


# Band cut filter 带宽过滤器
def butter_bandstop_filter(data, lowcut, highcut, fs, order):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq

    i, u = butter(order, [low, high], btype='bandstop')
    y = lfilter(i, u, data)
    return y


# Band cut filter, high pass
def butter_highpass_filter(data, cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='high')
    y = lfilter(b, a, data)
    return y


# Creation of the pointer to the patient file with index equal to index
# 创建指向患者文件的指针, 使index等于index.
def loadSummaryPatient(index):
    f = open(pathDataSet + 'chb' + patients[index] + '/chb' + patients[index] + '-summary.txt')
    return f


# Patient data loading(indexPatient).
# Data is taken from the file with the filename given in fileOfData.
# Returns a numpy array with patient data contained in the file.
# 患者数据加载(indexPatient)数据是从fileOfData中指定的文件中获取的.
# 返回一个numpy数组, 包含文件中包含的患者数据.
def loadDataOfPatient(indexPatient, fileOfData):
    f = pyedflib.EdfReader(pathDataSet + 'chb' + patients[
        indexPatient] + '/' + fileOfData)  # https://pyedflib.readthedocs.io/en/latest/#description
    n = f.signals_in_file
    sigbufs = np.zeros((n, f.getNSamples()[0]))
    for i in np.arange(n):
        sigbufs[i, :] = f.readSignal(i)
    sigbufs = cleanData(sigbufs, indexPatient)
    return sigbufs


def cleanData(Data, indexPatient):
    if patients[indexPatient] in ["19", "21"]:
        Data = np.delete(Data, 21, axis=0)
        Data = np.delete(Data, 17, axis=0)
        Data = np.delete(Data, 12, axis=0)
        Data = np.delete(Data, 9, axis=0)
        Data = np.delete(Data, 4, axis=0)
    return Data


# Conversion of a time string to a datetime type object and cleaning dates that do not respect time limits.
# 将时间字符串转换为datetime类型对象和不受时间限制的清洁日期.
def getTime(dateInString):
    try:
        time = datetime.strptime(dateInString, '%H:%M:%S')
    except ValueError:
        dateInString = " " + dateInString
        if ' 24' in dateInString:
            dateInString = dateInString.replace(' 24', '23')
            time = datetime.strptime(dateInString, '%H:%M:%S')
            time += timedelta(hours=1)
        else:
            dateInString = dateInString.replace(' 25', '23')
            time = datetime.strptime(dateInString, '%H:%M:%S')
            time += timedelta(hours=2)
    return time


def saveSignalsOnDisk(signalsBlock, nSpectogram):
    global SecondPartPathOutput
    global FirstPartPathOutput
    global legendOfOutput
    global isPreictal

    if not os.path.exists(FirstPartPathOutput):
        os.makedirs(FirstPartPathOutput)
    if not os.path.exists(FirstPartPathOutput + SecondPartPathOutput):
        os.makedirs(FirstPartPathOutput + SecondPartPathOutput)
    np.save(FirstPartPathOutput + SecondPartPathOutput + '/spec_' + isPreictal + '_' + str(
        nSpectogram - signalsBlock.shape[0]) + '_' + str(nSpectogram - 1), signalsBlock)
    legendOfOutput = legendOfOutput + str(nSpectogram - signalsBlock.shape[0]) + ' ' + str(
        nSpectogram - 1) + ' ' + SecondPartPathOutput + '/spec_' + isPreictal + '_' + str(
        nSpectogram - signalsBlock.shape[0]) + '_' + str(nSpectogram - 1) + '.npy\n'


# splits data contained in data into Windows and creates spectrograms that are saved on disk
# s is the factor indicating how far each window moves
# returns data that is not considered; this happens when the data is not divisible by the length of the window
# 将数据中包含的数据拆分到窗口中, 并创建保存在磁盘上的频谱图
# s是表示每个窗口移动多远的因子
# 返回未被考虑的数据, 当数据不能被窗口长度整除时就会发生这种情况.
def createSpectrogram(data, S=0):
    global nSpectrogram
    global signalsBlock
    global inB
    signals = np.zeros((22, 59, 114))

    t = 0
    movement = int(S * 256)
    if S == 0:
        movement = _SIZE_WINDOW_SPECTOGRAM
    while data.shape[1] - (t * movement + _SIZE_WINDOW_SPECTOGRAM) > 0:
        # creating the spectrogram for all channels
        for i in range(0, 22):
            start = t * movement
            stop = start + _SIZE_WINDOW_SPECTOGRAM
            signals[i, :] = createSpec(data[i, start:stop])
        if signalsBlock is None:
            signalsBlock = np.array([signals])
        else:
            signalsBlock = np.append(signalsBlock, [signals], axis=0)
        nSpectrogram = nSpectrogram + 1
        if signalsBlock.shape[0] == 50:
            saveSignalsOnDisk(signalsBlock, nSpectrogram)
            signalsBlock = None
            # saving signals
        t = t + 1
    return (data.shape[1] - t * _SIZE_WINDOW_SPECTOGRAM) * -1


# function for true spectrogram creation.
# 用于创建真实频谱图的函数。
def createSpec(data):
    fs = 256
    lowCut = 117
    highCut = 123

    y = butter_bandstop_filter(data, lowCut, highCut, fs, order=6)
    lowCut = 57
    highCut = 63
    y = butter_bandstop_filter(y, lowCut, highCut, fs, order=6)

    cutoff = 1
    y = butter_highpass_filter(y, cutoff, fs, order=6)

    Pxx = signal.spectrogram(y, nfft=256, fs=256, noverlap=128)[2]
    Pxx = np.delete(Pxx, np.s_[117:123 + 1], axis=0)
    Pxx = np.delete(Pxx, np.s_[57:63 + 1], axis=0)
    Pxx = np.delete(Pxx, 0, axis=0)

    result = (10 * np.log10(np.transpose(Pxx)) - (10 * np.log10(np.transpose(Pxx))).min()) / (
            10 * np.log10(np.transpose(Pxx))).ptp()
    return result


# Creating spectrogram and visualization with matplotlib library.
# 创建频谱图并使用matplotlib库进行可视化。
def createSpecAndPlot(data):
    freqs, bins, Pxx = signal.spectrogram(data, nfft=256, fs=256, noverlap=128)

    print("Original")
    plt.pcolormesh(freqs, bins, 10 * np.log10(np.transpose(Pxx)), cmap=plt.cm.jet)
    plt.colorbar()
    plt.ylabel('sec')
    plt.xlabel('Hz')
    plt.title('Spectrogram')
    plt.show()
    plt.close()

    fs = 256
    lowCut = 117
    highCut = 123

    y = butter_bandstop_filter(data, lowCut, highCut, fs, order=6)
    lowCut = 57
    highCut = 63
    y = butter_bandstop_filter(y, lowCut, highCut, fs, order=6)

    cutoff = 1
    y = butter_highpass_filter(y, cutoff, fs, order=6)

    # Pxx=signal.spectrogram(y, nfft=256, fs=256, return_onesided=True, noverlap=128)[2]
    freqs, bins, Pxx = signal.spectrogram(y, nfft=256, fs=256, noverlap=128)

    print("Filtered")
    plt.pcolormesh(freqs, bins, 10 * np.log10(np.transpose(Pxx)), cmap=plt.cm.jet)
    plt.colorbar()
    plt.ylabel('sec')
    plt.xlabel('Hz')
    plt.title('Spectrogram')
    plt.show()
    plt.close()

    Pxx = np.delete(Pxx, np.s_[117:123 + 1], axis=0)
    Pxx = np.delete(Pxx, np.s_[57:63 + 1], axis=0)
    Pxx = np.delete(Pxx, 0, axis=0)

    print("Cleaned but not standard")
    freqs = np.arange(Pxx.shape[0])
    plt.pcolormesh(freqs, bins, 10 * np.log10(np.transpose(Pxx)), cmap=plt.cm.jet)
    plt.colorbar()
    plt.ylabel('sec')
    plt.xlabel('Hz')
    plt.title('Spettrogramma')
    plt.show()
    plt.close()

    result = (10 * np.log10(np.transpose(Pxx)) - (10 * np.log10(np.transpose(Pxx))).min()) / (
            10 * np.log10(np.transpose(Pxx))).ptp()

    print("Standard")
    freqs = np.arange(result.shape[1])
    plt.pcolormesh(freqs, bins, result, cmap=plt.cm.jet)
    plt.colorbar()
    plt.ylabel('sec')
    plt.xlabel('Hz')
    plt.title('Spectrogram')
    plt.show()
    plt.close()

    return result


# Class used to represent ranges of data, both preictal and interictal.
# 用于表示数据范围的类, 无论是发作前还是发作间期.
class PreIntData:
    start = 0
    end = 0

    def __init__(self, s, e):
        self.start = s
        self.end = e


# Class used to keep file data, date and time start and end and associated file name.
# 用于保留文件数据, 日期和时间开始和结束以及关联的文件名的类.
class FileData:
    start = 0
    end = 0
    nameFile = ""

    def __init__(self, s, e, nF):
        self.start = s
        self.end = e
        self.nameFile = nF


# function that stores all useful data of the patient analysed
# pointer to the summary file of the analysed patient
# returns: preictalInterval: PreIntData vector with all ranges of all preictal data
#          interictalInterval: PreIntData vector with all ranges of all interictal data
#          files: filename vector with all the data of the various files
# 函数存储分析患者的所有有用数据
# 分析患者的摘要文件的指针
# 返回: preictalInterval: PreIntData向量, 其中包含所有发作前数据的所有范围
#      interictalInterval: PreIntData向量, 其中包含所有发作间期数据的所有范围
#      files: filename向量, 其中包含各个文件的所有数据
def createArrayIntervalData(fSummary):
    preictalInterval = []
    interictalInterval = [PreIntData(datetime.min, datetime.max)]
    files = []
    firstTime = True
    oldTime = datetime.min  # Equivalent of 0 on dates
    startTime = 0
    line = fSummary.readline()
    endS = datetime.min
    while line:
        data = line.split(':')
        if data[0] == "File Name":
            nF = data[1].strip()
            s = getTime((fSummary.readline().split(": "))[1].strip())
            if firstTime:
                interictalInterval[0].start = s
                firstTime = False
                startTime = s
            while s < oldTime:  # If it changes by day, add 24 hours to the date.
                s = s + timedelta(hours=24)
            oldTime = s
            endTimeFile = getTime((fSummary.readline().split(": "))[1].strip())
            while endTimeFile < oldTime:  # If it changes by day, add 24 hours to the date.
                endTimeFile = endTimeFile + timedelta(hours=24)
            oldTime = endTimeFile
            files.append(FileData(s, endTimeFile, nF))
            for j in range(0, int((fSummary.readline()).split(':')[1])):
                secSt = int(fSummary.readline().split(': ')[1].split(' ')[0])
                secEn = int(fSummary.readline().split(': ')[1].split(' ')[0])
                ss = s + timedelta(seconds=secSt) - timedelta(
                    minutes=_MINUTES_OF_DATA_BETWEEN_PRE_AND_SEIZURE + _MINUTES_OF_PREICTAL)
                if (len(preictalInterval) == 0 or ss > endS) and ss - startTime > timedelta(minutes=20):
                    ee = ss + timedelta(minutes=_MINUTES_OF_PREICTAL)
                    preictalInterval.append(PreIntData(ss, ee))
                endS = s + timedelta(seconds=secEn)
                ss = s + timedelta(seconds=secSt) - timedelta(hours=4)
                ee = s + timedelta(seconds=secEn) + timedelta(hours=4)
                if (interictalInterval[len(interictalInterval) - 1].start < ss and interictalInterval[
                    len(interictalInterval) - 1].end > ee):
                    interictalInterval[len(interictalInterval) - 1].end = ss
                    interictalInterval.append(PreIntData(ee, datetime.max))
                else:
                    if interictalInterval[len(interictalInterval) - 1].start < ee:
                        interictalInterval[len(interictalInterval) - 1].start = ee
        line = fSummary.readline()
    fSummary.close()
    interictalInterval[len(interictalInterval) - 1].end = endTimeFile
    return preictalInterval, interictalInterval, files


def main():
    global SecondPartPathOutput
    global FirstPartPathOutput
    global legendOfOutput
    global nSpectrogram
    global signalsBlock
    global isPreictal  # 是否为发作前
    print("START \n")
    loadParametersFromFile("PARAMETERS_DATA_EDITING.txt")
    print("Parameters loaded")

    for indexPatient in range(0, len(patients)):
        print("Working on patient " + patients[indexPatient])
        legendOfOutput = ""
        nSpectrogram = 0

        SecondPartPathOutput = '/paz' + patients[indexPatient]
        f = loadSummaryPatient(indexPatient)
        preictalInfo, interictalInfo, filesInfo = createArrayIntervalData(f)
        if patients[indexPatient] == "19":
            preictalInfo.pop(0)  # Deletion of data from the first indent because it is not considered
        print("Summary patient loaded")

        # Start of management cycle interms date
        print("START creation interictal spectrogram")
        totInst = 0
        # c=0
        # d=0   
        interictalData = np.array([]).reshape(22, 0)
        indexInterictalSegment = 0
        isPreictal = ''
        for fInfo in filesInfo:
            fileS = fInfo.start
            fileE = fInfo.end
            intSegStart = interictalInfo[indexInterictalSegment].start
            intSegEnd = interictalInfo[indexInterictalSegment].end
            while fileS > intSegEnd and indexInterictalSegment < len(interictalInfo):
                indexInterictalSegment = indexInterictalSegment + 1
                intSegStart = interictalInfo[indexInterictalSegment].start
                intSegEnd = interictalInfo[indexInterictalSegment].end
            if not fileE < intSegStart or fileS > intSegEnd:
                if fileS >= intSegStart:
                    start = 0
                else:
                    start = (intSegStart - fileS).seconds
                if fileE <= intSegEnd:
                    end = 0
                else:
                    end = (intSegEnd - fileS).seconds
                tmpData = loadDataOfPatient(indexPatient, fInfo.nameFile)
                if not end == 0:
                    end = end * 256
                if tmpData.shape[0] < 22:
                    print(patients[indexPatient] + "Fewer channels, do not consider the file " + fInfo.nameFile)
                else:
                    interictalData = np.concatenate((interictalData, tmpData[0:22, start * 256:end]), axis=1)
                    notUsed = createSpectrogram(interictalData)
                    totInst += interictalData.shape[1] / 256 - notUsed / 256
                    interictalData = np.delete(interictalData, np.s_[0:interictalData.shape[1] - notUsed], axis=1)

        # Window_size: length_data_i = s :(length_data_p-30_sec_for _each aspect)
        if totInst == 0:
            S = 0
        else:
            S = (_SIZE_WINDOW_IN_SECONDS * (
                    len(preictalInfo) * _MINUTES_OF_PREICTAL * 60 - _SIZE_WINDOW_IN_SECONDS *
                    len(preictalInfo))) / totInst
        if not (signalsBlock is None):
            saveSignalsOnDisk(signalsBlock, nSpectrogram)
        signalsBlock = None

        print("Spectrogram interictal: " + str(nSpectrogram))
        print("Hours interictal: " + str(totInst / 60 / 60))
        legendOfOutput = str(nSpectrogram) + "\n" + legendOfOutput
        legendOfOutput = "INTERICTAL" + "\n" + legendOfOutput
        legendOfOutput = "SEIZURE: " + str(len(preictalInfo)) + "\n" + legendOfOutput
        legendOfOutput = patients[indexPatient] + "\n" + legendOfOutput
        allLegend = legendOfOutput
        legendOfOutput = ''
        nSpectrogram = 0
        print("END creation interictal spectrogram")

        print("START creation preictal spectrogram")
        isPreictal = 'P'
        contSeizure = -1
        for pInfo in preictalInfo:
            contSeizure = contSeizure + 1
            legendOfOutput = legendOfOutput + "SEIZURE " + str(contSeizure) + "\n"
            preictalData = np.array([]).reshape(22, 0)
            j = 0
            for j in range(0, len(filesInfo)):
                if filesInfo[j].start <= pInfo.start < filesInfo[j].end:
                    break
            start = (pInfo.start - filesInfo[j].start).seconds
            if start < 0:
                start = 0  # if preictal starts before the file starts
            if pInfo.end <= filesInfo[j].end:
                end = (pInfo.end - filesInfo[j].start).seconds
                tmpData = loadDataOfPatient(indexPatient, filesInfo[j].nameFile)
                preictalData = np.concatenate((preictalData, tmpData[0:22, start * 256:end * 256]), axis=1)
            else:
                tmpData = loadDataOfPatient(indexPatient, filesInfo[j].nameFile)
                preictalData = np.concatenate((preictalData, tmpData[0:22, start * 256:]), axis=1)
                end = (pInfo.end - filesInfo[j + 1].start).seconds
                tmpData = loadDataOfPatient(indexPatient, filesInfo[j + 1].nameFile)
                preictalData = np.concatenate((preictalData, tmpData[0:22, 0:end * 256]), axis=1)
            createSpectrogram(preictalData, S=S)
            if not (signalsBlock is None):
                saveSignalsOnDisk(signalsBlock, nSpectrogram)
            signalsBlock = None

        allLegend = allLegend + "\n" + "PREICTAL" + "\n" + str(nSpectrogram) + "\n" + legendOfOutput
        print("Spectrogram preictal: " + str(nSpectrogram))
        print("SEIZURE: " + str(len(preictalInfo)))
        print("END creation preictal spectrogram")
        # END cycle management preictal data

        # START cycle management preictal data
        print("START creation \'real\' preictal spectrogram")
        isPreictal = 'P_R'
        nSpectrogram = 0
        contSeizure = -1
        S = 0
        legendOfOutput = ''
        for pInfo in preictalInfo:
            contSeizure = contSeizure + 1
            legendOfOutput = legendOfOutput + "SEIZURE " + str(contSeizure) + "\n"
            preictalData = np.array([]).reshape(22, 0)
            j = 0
            for j in range(0, len(filesInfo)):
                if filesInfo[j].start <= pInfo.start < filesInfo[j].end:
                    break
            start = (pInfo.start - filesInfo[j].start).seconds
            if start < 0:
                start = 0  # if preictal starts before the file starts
            if pInfo.end <= filesInfo[j].end:
                end = (pInfo.end - filesInfo[j].start).seconds
                tmpData = loadDataOfPatient(indexPatient, filesInfo[j].nameFile)
                preictalData = np.concatenate((preictalData, tmpData[0:22, start * 256:end * 256]), axis=1)
            else:
                tmpData = loadDataOfPatient(indexPatient, filesInfo[j].nameFile)
                preictalData = np.concatenate((preictalData, tmpData[0:22, start * 256:]), axis=1)
                end = (pInfo.end - filesInfo[j + 1].start).seconds
                tmpData = loadDataOfPatient(indexPatient, filesInfo[j + 1].nameFile)
                preictalData = np.concatenate((preictalData, tmpData[0:22, 0:end * 256]), axis=1)
            createSpectrogram(preictalData, S=S)
            if not (signalsBlock is None):
                saveSignalsOnDisk(signalsBlock, nSpectrogram)
            signalsBlock = None

        allLegend = allLegend + "\n" + "REAL_PREICTAL" + "\n" + str(nSpectrogram) + "\n" + legendOfOutput
        print("Spectrogram \'REAL\' preictal: " + str(nSpectrogram))
        print("END creation \'real\' preictal spectrogram")

        text_file = open(FirstPartPathOutput + SecondPartPathOutput + "/legendAllData.txt", "w")
        text_file.write(allLegend)
        text_file.close()
        print("Legend saved on disk")
        print('\n')
    print("END")


if __name__ == '__main__':
    main()
