import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np

# Color map for plots, and global graph defaults
colormap = np.array(['red', 'green', 'blue', 'black', 'darkred', 'coral', 'peru', 'darkorange', 'olive', 'yellow', 'lawngreen', 'lime', 'aquamarine', 'cyan', 'steelblue', 'navy', 'indigo', 'violet', 'purple', 'fuchsia'])
plt.rcParams.update({'font.size': 16,
                     'axes.titlesize': 28, 'axes.titleweight': 'bold',
                     'axes.labelsize': 20, 'axes.labelweight': 'bold', 'xtick.labelsize': 16, 'ytick.labelsize': 16, 'lines.linewidth': 2.5})
baseLineCurrent = -1.020418863242887e-10 #measured on 06/20/25

# Returns data in x, y, z, V1, V2, I1, I2, I3, FT, line, LA, LP, dt
def convertCVData(run):
    print("Working..")
    with open(run) as f:
        data = [line.strip().split('\t') for line in f]
    # X (um) 0
    # Y (um) 1
    # Z (um) 2
    # V1 (V) 3
    # V2 (V) 4
    # Current 1 (A) 5
    # Current 2 (A) 6
    # Current 3 (A) 7
    # FeedbackType 8
    # Line Number 9
    # Lock in Amplitude 10
    # Lock in  Phase 11
    # dt(s) 12
    print("Working....")
    return np.array(data[0][0:-1], dtype=float), np.array(data[1][0:-1], dtype=float), np.array(data[2][0:-1], dtype=float), np.array(data[3][0:-1], dtype=float), np.array(data[4][0:-1], dtype=float), np.array(data[5][0:-1], dtype=float), np.array(data[6][0:-1], dtype=float), np.array(data[7][0:-1], dtype=float), np.array(data[8][0:-1], dtype=float), np.array(data[9][0:-1], dtype=float), np.array(data[10][0:-1], dtype=float), np.array(data[11][0:-1], dtype=float), np.array(data[12][0:-1], dtype=float)

# Prints the position assuming no scan hopping is done
def printPositionData(data):
    x, y, z, V, VV, I, II, III, FT, line, LA, LP, dt = convertCVData(data)
    position = len(z)-1
    print('X:' + str(x[position]))
    print('Y:' + str(y[position]))
    print('Z:' + str(z[position]))

# Returns a smooth dataset after FFT filter of f
def reduceNoiseOfData(data, f):
    N = len(data)
    fftData = np.fft.fft(data)
    freq = np.fft.fftfreq(N, d=1/f)
    mask = np.abs(freq) <= 1
    returnData = np.fft.ifft(fftData*mask)

    return np.array(returnData)

# Returns dI and cropped area of RuHex (assuming there is a RuHex wave and the scan goes over 0V [no reaction at 0V assumption])
def getSteadyState(I, V):
    temp = np.where((V >= -0.01) & (V <= 0.01))[0]
    # Positive Scan
    if I[temp[0]] > I[temp[-1]]:
        start = temp[-1]
    # Negative Scan
    else:
        start = temp[0]
    end = np.argmin(V[start:]) + start
    if end == start:
        end += 1

    returnI = I[start:end]
    returnV = V[start:end]

    dI = abs(returnI.min()-returnI.max())

    return dI, returnI, returnV

# Returns the baseline current and top and bottom capacitive (assumes the scan goes over 0V [no reaction at 0V assumption])
def getBaselineAndCharging(I, V):
    temp = np.where((V >= -0.1) & (V <= 0.1))[0]
    # Positive Scan
    if I[temp[0]] > I[temp[-1]]:
        topI = I[temp[0]:int(0.5*len(I)-1)].min()
        botI = I[temp[int(0.5*len(temp)+1)]:].max()
    # Negative Scan
    else:
        topI = I[int(0.5*len(I)+1):temp[-1]].min()
        botI = I[:temp[int(0.5*len(temp)-1)]].max()

    #baseLineCurrent = botI

    return baseLineCurrent, topI, botI

# Returns indexes where transitions occur
def findTransitions(lines):
    dline = np.diff(lines)
    transitions = np.where(dline == 1)[0] + 1
    return transitions

# Returns the data split at transitions
def splitData(data, transitions):
    Data = [data[:int(transitions[0])]]
    for i in range(0, len(transitions) - 1):
        Data.append(data[int(transitions[i]):int(transitions[i+1])])
    Data.append(data[int(transitions[-1]):])
    return np.array(Data, dtype=object)

# Returns the data grouped every group amount
def groupData(data, group):
    Data = []
    for i in range(0, len(data), group):
        Data.append(np.concatenate(data[i:i+group]))

    return np.array(Data, dtype=object)

# Returns the charge of an oxidation peak (many assumpitons)
def integrateCharge(IData, t, VData, epsilon=10**-14):
    QData = []
    iss = []
    timeIndex = 0

    for d in range(0, len(IData)):
        I = IData[d]
        V = VData[d]
        T = t[timeIndex:timeIndex + len(I)]
        halfWay = int(len(I)/2)

        baseLineCurrent, topI, botI = getBaselineAndCharging(I, V)

        a = np.where(np.abs(I[:halfWay]-topI) < epsilon)[0]
        b = np.where(np.abs(I[halfWay:]-botI) < epsilon)[0]

        if a.size == 0:
            a = halfWay
        else:
            a = a[-1]

        if b.size == 0:
            b = halfWay
        else:
            b = b[0] + halfWay

        timeSegment = T[a:b]
        ISegment = I[a:b]
        m = (I[b]-I[a])/(T[b]-T[a])
        inter = I[a] - m*T[a]
        line = m*np.array(timeSegment) + inter

        TotalCharge = np.trapezoid(ISegment, timeSegment)
        CapacitiveCharge = np.trapezoid(line, timeSegment)

        QData.append((TotalCharge-CapacitiveCharge))

        dI, *_ = getSteadyState(I, V)
        iss.append(dI)

        # plt.figure(figsize=(15, 9))
        # plt.plot(T, I)
        # plt.plot(timeSegment, line, color='orange')
        # plt.plot([T[a], T[b]], [I[a], I[b]], color='orange')
        # plt.title('')
        # plt.xlabel('Time (s)')
        # plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda val, _: f"{(val - baseLineCurrent) * 1e12:.1f}"))
        # plt.yticks(np.unique(np.append(plt.yticks()[0], baseLineCurrent)))
        # plt.axhline(baseLineCurrent, color='gray', linestyle='dashed', linewidth=1)
        # plt.ylabel('Current (pA)')
        # plt.grid(True)
        # plt.show()

        timeIndex += len(I)

    return QData, iss

# Returns the charge, current, and time of each IData and plots ITs and VTs (legacy use of charge intagration, many asuumptions)
def CVtoITandVT(IData, VData, dt, cutTime, baseLineCurrent, name1='', name2=''):
    time = [0]
    for t in dt:
        time.append((t+time[-1]))
    time = time[:-cutTime+1]

    QData, iss = integrateCharge(IData, time, VData)

    voltage = np.concatenate(VData)
    current = np.concatenate(IData)

    # Linear sweep of first
    plt.figure(figsize=(15, 9))
    plt.plot(time[:len(VData[0])], VData[0], 'k-')
    plt.title(name1)
    plt.xlabel('Time (s)')
    plt.ylabel('Potential (V vs. Ag/AgCl)')
    plt.grid(True)

    # Linear sweep of all
    plt.figure(figsize=(15, 9))
    plt.plot(time, voltage, 'k-')
    plt.title(name1)
    plt.xlabel('Time (s)')
    plt.ylabel('Potential (V vs. Ag/AgCl)')
    plt.grid(True)

    # Current time trace
    plt.figure(figsize=(15, 9))
    plt.plot(time, current)
    plt.title(name2)
    plt.xlabel('Time (s)')
    plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda val, _: f"{(val - baseLineCurrent) * 1e12:.1f}"))
    plt.yticks(np.unique(np.append(plt.yticks()[0], baseLineCurrent)))
    plt.axhline(baseLineCurrent, color='gray', linestyle='dashed', linewidth=1)
    plt.ylabel('Current (pA)')
    plt.grid(True)

    return QData, iss, time

# Returns smoothed Current, Volts, and plots CV data
def plotCV(pathName, findSteadyState=False, quietTimeCut=True, f=120, name=''):
    x, y, z, V, VV, I, II, III, FT, line, LA, LP, dt = convertCVData(pathName)
    I1 = reduceNoiseOfData(I, f)

    print('Working......')

    #baseLineCurrent, *_ = getBaselineAndCharging(I1, V)

    if quietTimeCut:
        transitions = findTransitions(line)
        s = transitions[1]
        V = V[s:]
        I = I[s:]
        I1 = I1[s:]

    plt.figure(figsize=(15, 9))
    plt.plot(V, I, 'k-')
    plt.plot(V, I1, 'b-')
    if findSteadyState:
        dI, I2, *_ = getSteadyState(I1, V)
        print("The steady state current was found to be: " + str(dI) + " amps.")
        plt.axhline(y=I2.max(), color='g', linestyle='--')
        plt.axhline(y=I2.min(), color='g', linestyle='--')
    plt.title(name)
    plt.xlabel('Potential (V vs. Ag/AgCl)')
    plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda val, _: f"{(val - baseLineCurrent) * 1e12:.1f}"))
    if findSteadyState:
        plt.yticks(np.unique(np.append(plt.yticks()[0], baseLineCurrent)))
        plt.axhline(baseLineCurrent, color='gray', linestyle='dashed', linewidth=1)
    plt.ylabel('Current (pA)')
    plt.grid(True)
    # plt.show()

    return I1, V, dt, I

# Returns smoothed Current, Time, and plots CV data
def plotIT(pathName, quietTimeCut=True, f=120,  name=''):
    x, y, z, V, VV, I, II, III, FT, line, LA, LP, dt = convertCVData(pathName)
    I1 = reduceNoiseOfData(I, f)

    print("Working......")
    print((dt[10]))

    t = [0]
    for i in range(int(np.array(dt).size) - 1):
        t.append(t[i]+dt[i]+0)

    t = np.array(t)

    print('Working........')

    if quietTimeCut:
        transitions = findTransitions(line)
        s = transitions[1]
        t = t[s:]
        I = I[s:]
        I1 = I1[s:]

    plt.figure(figsize=(15, 9))
    plt.plot(t, I, 'k-')
    plt.plot(t, I1, 'b-')
    plt.title(name)
    plt.xlabel('Time (seconds)')
    plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda val, _: f"{val * 1e12:.1f}"))
    plt.ylabel('Current (pA)')
    plt.grid(True)
    # plt.show()

    return I1, t, I

# Returns hight, smoothed Current, unsmoothed current, Time, and plots approach data
def plotApproach(pathName, f=120,  name=''):
    x, y, z, V, VV, I, II, III, FT, line, LA, LP, dt = convertCVData(pathName)
    I1 = reduceNoiseOfData(I, f)

    t = np.array([0])
    for i in range(int(np.array(dt).size)):
        t = np.append(t, t[i]+dt[i])

    plt.figure(figsize=(15, 9))
    plt.plot(z, I, 'k-')
    plt.plot(z, I1, 'b-')
    plt.title(name)
    plt.xlabel('Distance (micro-m)')
    plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda val, _: f"{val * 1e12:.1f}"))
    plt.ylabel('Current (pA)')
    plt.legend(['Data', 'RunningAverage'], fontsize=14)
    plt.grid(True)

    plt.figure(figsize=(15, 9))
    plt.plot(t[0:-1], I, 'k-')
    plt.plot(t[0:-1], I1, 'b-')
    plt.title('Time vs Current')
    plt.xlabel('Time (s)')
    plt.ylabel('Current (pA)')
    plt.legend(['Data', 'RunningAverage'], fontsize=14)
    plt.grid(True)

    plt.figure(figsize=(15, 9))
    plt.plot(t[0:-1], z, 'k-')
    plt.title('Time vs Distance')
    plt.xlabel('Time (s)')
    plt.ylabel('Distance (micro-m)')
    plt.grid(True)
    #plt.show()

    return z, I1, I, t[0:-1]

# Plots scan hopping graphs
def plotScanHopping(pathName, graphXY=False, graphSteadyState=False,  offset=7, steadyStateName='Steady State Current', oneGraph=False, f=120, name='Current vs Voltage'):
    x, y, z, V, VV, I, II, III, FT, line, LA, LP, dt = convertCVData(pathName)
    transitions = findTransitions(line)

    VData = splitData(V, transitions)
    IData = splitData(I, transitions)
    xData = splitData(x, transitions)
    yData = splitData(y, transitions)

    # assumptions about grouping of the data, often used as a guideline, not truth
    print("The number scans done is: " + str(int((len(VData)+offset-10)/8)) + ".")

    if oneGraph:
        legend = np.array([])
        plt.figure(figsize=(15, 9))
        for j in range(offset, len(VData), 8):
            voltage = np.append(np.append(VData[j], VData[j+1]), VData[j+2])
            current = reduceNoiseOfData(np.append(np.append(IData[j], IData[j+1]), IData[j+2]), f)
            plt.plot(voltage, current, '-', color=colormap[int((j-offset)/8)])
            legend = np.append(legend, ('Position X:' + str(int(xData[j][10])) + ', Y:' + str(int(yData[j][10]))))
        plt.title(name)
        plt.xlabel('Voltage (Volts)')
        plt.ylabel('Current (Amp)')
        plt.legend(legend, fontsize=14)
        plt.grid(True)
    else:
        steadyStateCurrent = np.array([])
        runPos = np.array([])
        for j in range(offset, len(VData), 8):
            voltage = np.append(np.append(VData[j], VData[j+1]), VData[j+2])
            current1 = np.append(np.append(IData[j], IData[j+1]), IData[j+2])
            current2 = reduceNoiseOfData(np.append(np.append(IData[j], IData[j+1]), IData[j+2]), f)
            runPos = np.append(runPos, (str(int(xData[j][10])) + ':' + str(int(yData[j][10]))))

            legend = np.array(['Data', (runPos[int((j-offset)/8)] + ' (μm)')])
            plt.figure(figsize=(15, 9))
            plt.plot(voltage, current1, 'k-')
            plt.plot(voltage, current2, 'b-')
            if graphSteadyState:
                dI, I2, *_ = getSteadyState(current2, voltage)
                steadyStateCurrent = np.append(steadyStateCurrent, dI)
                plt.axhline(y=I2.max(), color='g', linestyle='--')
                plt.axhline(y=I2.min(), color='g', linestyle='--')
                legend = np.append(legend, ('ΔI: ' + str("%.2f" % (dI*10**12)) + ' (pico-amps)'))
            plt.title(name)
            plt.xlabel('Voltage (Volts)')
            plt.ylabel('Current (Amp)')
            plt.legend(legend, fontsize=14)
            plt.grid(True)

        print('Working......')

        if graphSteadyState:
            plt.figure(figsize=(15, 9))
            plt.bar(runPos, steadyStateCurrent)
            plt.title(steadyStateName)
            plt.xlabel('Position (X:Y) (μm)')
            plt.ylabel('Current (Amp)')

        if graphXY:
            plt.figure(figsize=(15, 9))
            x = np.array([])
            y = np.array([])
            for k in runPos:
                x = np.append(x, int(k.split(':')[0]))
                y = np.append(y, int(k.split(':')[1]))
            plt.scatter(x, y, color='k')
            plt.title('X vs Y Position')
            plt.xlabel('X (μm)')
            plt.ylabel('Y (μm)')
            plt.grid(True)

    # plt.show()

# Plots CV of cycled data (legacy use of charge intagration, many asuumptions)
def plotCyclingCV(pathName, cycleStart=0, findSteadyState=False, quietTimeCut=True, haveLegend=True, goTill=0, checkEachCV=False, ConvertCVtoIT=False, f=120,  name='', grouping=1, mergeCluster=None):
    x, y, z, V, VV, I, II, III, FT, line, LA, LP, dt = convertCVData(pathName)
    transitions = findTransitions(line)

    VData = splitData(V, transitions)
    IData = splitData(I, transitions)
    returnV = []
    returnI = []
    QData = []
    iss = []
    time = []

    offset = cycleStart
    if quietTimeCut:
        offset += 1

    if grouping > 1:
        if quietTimeCut:
            VData = np.array([VData[0]] + list(groupData(VData[1:], grouping)), dtype=object)
            IData = np.array([IData[0]] + list(groupData(IData[1:], grouping)), dtype=object)
        else:
            VData = groupData(VData, grouping)
            IData = groupData(IData, grouping)

    mergeLength = 0
    if mergeCluster:
        mergeLength = len(mergeCluster)
    print("The number of cycles is: " + str(len(VData)-2-cycleStart-mergeLength) + ".")
    print('Working......')


    legend = [('Cycle: ' + str(offset-cycleStart))]

    voltage = np.append(VData[offset], VData[offset+1][:int(len(VData[offset+1])/2)])
    current = reduceNoiseOfData(np.append(IData[offset], IData[offset+1][:int(len(VData[offset+1])/2)]), f)
    # plt.plot(voltage, current, '-', color=colormap[(offset-cycleStart) % len(colormap)])
    returnV.append(voltage)
    returnI.append(current)

    if findSteadyState:
        dI, *_ = getSteadyState(current, voltage)
        print("The steady state current was found to be: " + str(dI) + " amps.")

    if goTill == 0:
        goTill = len(VData)-1
    elif goTill > len(VData):
        print("Error: Too many cycles")
        return

    for j in range(offset+1, goTill-1):
        firstIndex = int(len(VData[j])/2)
        secondIndex = int(len(VData[j+1])/2)
        voltage = np.append(VData[j][firstIndex:], VData[j+1][:secondIndex])
        current = reduceNoiseOfData(np.append(IData[j][firstIndex:], IData[j+1][:secondIndex]), f)
        # plt.plot(voltage, current, '-', color=colormap[j % len(colormap)])
        returnV.append(voltage)
        returnI.append(current)
        legend.append(('Cycle: ' + str(j-cycleStart)))

    if goTill == len(VData)-1:
        voltage = np.append(VData[-2][int(len(VData[-2])/2):], VData[-2+1])
        current = reduceNoiseOfData(np.append(IData[-2][int(len(VData[-2])/2):], IData[-2+1]), f)
        # plt.plot(voltage, current, '-', color=colormap[(len(VData)-2) % len(colormap)])
        returnV.append(voltage)
        returnI.append(current)
        legend.append(('Cycle: ' + str(len(VData)-2-cycleStart)))

    if mergeCluster:
        lastIndex = 0
        newReturnV = []
        newReturnI = []
        for r in mergeCluster:
            start, end = r[0], r[-1]
            newReturnV.extend(returnV[lastIndex:start])
            newReturnI.extend(returnI[lastIndex:start])

            mergedV = []
            mergedI = []
            for idx in r:
                mergedV += returnV[idx].tolist()
                mergedI += returnI[idx].tolist()
            newReturnV.append(np.array(mergedV))
            newReturnI.append(np.array(mergedI))

            lastIndex = end + 1
        newReturnV.extend(returnV[lastIndex:])
        newReturnI.extend(returnI[lastIndex:])

        returnV = np.array(newReturnV, dtype=object)
        returnI = np.array(newReturnI, dtype=object)

    # baseLineCurrent, *_ = getBaselineAndCharging(returnI[0], returnV[0])

    plt.figure(figsize=(15, 9))
    for i in range(0, len(returnV)):
        plt.plot(returnV[i], returnI[i], '-', color=colormap[i % len(colormap)])
    plt.title(name)
    plt.xlabel('Potential (V vs. Ag/AgCl)')
    plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda val, _: f"{(val - baseLineCurrent) * 1e12:.1f}"))
    if findSteadyState:
        plt.yticks(np.unique(np.append(plt.yticks()[0], baseLineCurrent)))
        plt.axhline(baseLineCurrent, color='gray', linestyle='dashed', linewidth=1)
    plt.ylabel('Current (pA)')
    if haveLegend:
        plt.legend(legend, fontsize=7)
    plt.grid(True)

    if checkEachCV:
        for i in range(0, len(returnI)):
            plt.figure()
            print(i+1)
            plt.plot(returnV[i], returnI[i])
            plt.show()

    print('Working........')

    if ConvertCVtoIT:
        cutTime = 2
        for o in range(0, offset):
            cutTime += len(VData[o])
        QData, iss, time = CVtoITandVT(returnI, returnV, dt, cutTime, baseLineCurrent)
        # constant = np.array(QData)/np.array(iss)
        # print(QData)
        # print(constant)

    # plt.show()

    return returnI, returnV, QData, iss, time

### Graphing of data goes here ###
### Baseline current is calibrated to H. Whites Group SECCM ###