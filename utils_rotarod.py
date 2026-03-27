import ruptures as rpt
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os


def rodSpeed_smoothing(rodSpeed_input, label, savefigpath):
    """smoothe the raw rotarod signal, 
      determine the start and end of the trial
      output:
        saveData: a dictionary containing raw and smoothed rod speed, time, start and run time
    """     
    #voltage linearly correlated with rod speed       
    startVoltage = [4.45, 40] # 4.45 V - 40 RPM 
    endVoltage = [8.90, 80]    # 8.90 V - 80 RPM
    rod_a = (endVoltage[1] - startVoltage[1])/(endVoltage[0] - startVoltage[0])
    rod_b = startVoltage[1] - rod_a*startVoltage[0]

    signal = rodSpeed_input[0].values/100

                # downsample it

    algo = rpt.Pelt(model="l2").fit(signal)
    # Predict the change points
    predicted_bkps = algo.predict(pen=3)

    # Display results
    # change the signal to rod speed

    rodSpeed = signal*rod_a+rod_b

    # Smooth the signal
    smoothed_signal = np.copy(rodSpeed)
    #running average first
    tempSpeed = rodSpeed
    #plt.figure()
    #plt.plot(rodSpeed)
    if rodSpeed[0] == 0 and rodSpeed[-1] == 0:
        # steady state has been recorded
        startRange = predicted_bkps[0]
        endRange = predicted_bkps[-2]
    elif rodSpeed[0] == 0 and rodSpeed[-1] > 0:
        startRange = predicted_bkps[0]
        endRange = len(rodSpeed)
    elif  rodSpeed[0] > 0 and rodSpeed[-1] == 0:
        # steady state hasn't been recorded
        startRange = 0
        endRange = predicted_bkps[-2]
    else:
        startRange = 0
        endRange = len(rodSpeed)

    for sss in range(0,60,10):
        windowSize = 4+sss*1

        for i in range(startRange, endRange):
            if i > startRange + windowSize/2 and i < endRange - windowSize/2:
                smoothed_signal[i] = np.mean(tempSpeed[i-windowSize//2 : i+windowSize//2])
            elif i <= startRange + windowSize/2:
                smoothed_signal[i] = np.mean(tempSpeed[startRange+1: i + windowSize // 2])
            elif i >= endRange - windowSize/2:
                smoothed_signal[i] = np.mean(tempSpeed[i - windowSize // 2: endRange-1])
        tempSpeed = smoothed_signal
        #plt.plot(tempSpeed)
        # jump = smoothed_signal[i] - smoothed_signal[i - 1]
        # if abs(jump) > max_jump:
        #     smoothed_signal[i] = smoothed_signal[i - 1] + max_jump
            #smoothed_signal[i] = (rodSpeed[i-1]+rodSpeed[i+1])/2

    algo = rpt.Pelt(model="l2").fit(smoothed_signal)
    # Predict the change points
    new_predicted_bkps = algo.predict(pen=1)

    # smooth one more round with the new predicted change point
    #plt.figure()
    #plt.plot(smoothed_signal)
    for sss in range(0,60,10):
        windowSize = 4+sss*1
        for i in range(startRange, endRange):
            if i > startRange + windowSize/2 and i < endRange - windowSize/2:
                smoothed_signal[i] = np.mean(tempSpeed[i-windowSize//2 : i+windowSize//2])
            elif i <= startRange + windowSize/2:
                smoothed_signal[i] = np.mean(tempSpeed[startRange+1: i + windowSize // 2])
            elif i >= endRange - windowSize/2:
                smoothed_signal[i] = np.mean(tempSpeed[i - windowSize // 2: endRange-1])

        # jump = smoothed_signal[i] - smoothed_signal[i - 1]
        # if abs(jump) > max_jump:
        #     smoothed_signal[i] = smoothed_signal[i - 1] + max_jump
            #smoothed_signal[i] = (rodSpeed[i-1]+rodSpeed[i+1])/2
    rodTime = rodSpeed_input[1].values/1000

    # algo = rpt.Pelt(model="l2").fit(smoothed_signal)
    # # Predict the change points
    # new_predicted_bkps = algo.predict(pen=170)

    # find the point when rod speed start to increase with first derivative
    dx = np.diff(smoothed_signal)/np.diff(rodTime)
    k=200
    runIdx = np.where(np.convolve((dx > 0.1).astype(int), np.ones(k, dtype=int), 'valid') == k)[0][0]

    startIdx = np.where(smoothed_signal>0.5)[0][0]
    endIdx = np.where(smoothed_signal>0.5)[0][-1]
    # save the running speed and voltage
    # save the smoothed_signal somewhere
    saveData={}
    saveData['raw'] = signal
    saveData['smoothed'] = smoothed_signal
    saveData['time'] = rodTime

    if rodSpeed[0] ==0:
        saveData['Start'] = np.zeros((len(rodTime)))+rodTime[startIdx]
        saveData['Run'] = np.zeros((len(rodTime)))+rodTime[runIdx]
    else:
        saveData['Start'] = np.full((len(rodTime)),np.nan)
        saveData['Run'] = np.zeros((len(rodTime)))+rodTime[runIdx]
    saveData['time0'] = saveData['Run']
    saveData['time_aligned'] = saveData['time'] - saveData['time0']
    
    # save the smoothed rodspeed 
    if not os.path.exists(savefigpath):
        os.makedirs(savefigpath)
    savefigname = os.path.join(savefigpath, 'rodSpeed_smoothed.png')

    plt.figure()
    plt.plot(rodTime,rodSpeed)
    plt.plot(rodTime,smoothed_signal)
    plt.scatter(rodTime[startIdx], 0, s=200)
    plt.scatter(rodTime[runIdx], 0, s=200)
    plt.scatter(rodTime[endIdx],0, s=200)
    plt.title('Start point for ' + label)
    #plt.show()
    plt.savefig(savefigname)
    plt.close()

    return saveData



