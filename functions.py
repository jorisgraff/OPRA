import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras import Input, Model

import tensorflow.keras
from tensorflow.keras.models import model_from_json

import numpy as np
import pandas as pd

from tcn import TCN

import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch
import matplotlib._color_data as mcd
import math

def generate_signals(iteration,nsignals):
    if nsignals > 5:
        print("Too many signals")
        return None
    if iteration > 2:
        print("Too many iterations")
        return None
    if nsignals > 0:
        if iteration == 1:
            sine = list(range(10000))
            sine = [math.sin(math.pi * x / 500) for x in sine]
            sine[3209:3612] = [-0.3] * 403
            sine[6792:7150] = [-0.3] * 358
            sine[8523:8910] = [-0.3] * 387

            signal1 = np.asarray(sine)
        if iteration == 2:
            sine = list(range(10000))
            sine = [math.sin(math.pi * x / 500) for x in sine]
            sine[1360:1733] = [-0.3] * 373
            sine[4306:4652] = [-0.3] * 346
            sine[8203:8619] = [-0.3] * 416
            sine[9699:10000] = [-0.3] * 301
            signal1 = np.asarray(sine)

    if nsignals > 1:
        if iteration == 1:
            signal2 = np.empty(shape=(10000))
            firstpart = list(range(2909))
            firstpart = [math.sin(math.pi * x / (2909)) for x in firstpart]
            signal2[0:2909] = firstpart
            secondpart = list(range(600))
            secondpart = [math.sin(math.pi * x / 600) * -1 for x in secondpart]
            signal2[2909:3509] = secondpart
            thirdpart = list(range(2983))
            thirdpart = [math.sin(math.pi * x / (2983)) for x in thirdpart]
            signal2[3509:6492] = thirdpart
            fourthpart = list(range(600))
            fourthpart = [math.sin(math.pi * x / 600) * -1 for x in fourthpart]
            signal2[6492:7092] = fourthpart
            fifthpart = list(range(1131))
            fifthpart = [math.sin(math.pi * x / 1131) for x in fifthpart]
            signal2[7092:8223] = fifthpart
            sixthpart = list(range(600))
            sixthpart = [math.sin(math.pi * x / 600) * -1 for x in sixthpart]
            signal2[8223:8823] = sixthpart
            seventhpart = list(range(1177))
            seventhpart = [math.sin(math.pi * x / 1177) for x in seventhpart]
            signal2[8823:10000] = seventhpart
        if iteration == 2:
            signal2 = np.empty(shape=(10000))
            firstpart = list(range(1060))
            firstpart = [math.sin(math.pi * x / (1060)) for x in firstpart]
            signal2[0:1060] = firstpart
            secondpart = list(range(600))
            secondpart = [math.sin(math.pi * x / 600) * -1 for x in secondpart]
            signal2[1060:1660] = secondpart
            thirdpart = list(range(2346))
            thirdpart = [math.sin(math.pi * x / (2346)) for x in thirdpart]
            signal2[1660:4006] = thirdpart
            fourthpart = list(range(600))
            fourthpart = [math.sin(math.pi * x / 600) * -1 for x in fourthpart]
            signal2[4006:4606] = fourthpart
            fifthpart = list(range(3297))
            fifthpart = [math.sin(math.pi * x / 3297) for x in fifthpart]
            signal2[4606:7903] = fifthpart
            sixthpart = list(range(600))
            sixthpart = [math.sin(math.pi * x / 600) * -1 for x in sixthpart]
            signal2[7903:8503] = sixthpart
            seventhpart = list(range(895))
            seventhpart = [math.sin(math.pi * x / 895) for x in seventhpart]
            signal2[8503:9398] = seventhpart
            eighthpart = list(range(602))
            eighthpart = [math.sin(math.pi * x / 602) * -1 for x in eighthpart]
            signal2[9398:10000] = eighthpart

    if nsignals > 2:
        signal3 = np.empty(shape=(10000))
        i = 0
        while i < 10000:
            direction = np.random.randint(2)
            length = np.random.randint(300, 600)
            new = list(range(2 * length))
            if direction == 1:
                new = [math.sin(math.pi * x / length - 0.5 * math.pi) * 0.5 + 0.5 for x in new]
            else:
                new = [math.sin(math.pi * x / length + 0.5 * math.pi) * 0.5 - 0.5 for x in new]
            if i + 2 * length > 10000:
                signal3[i:10000] = new[0:10000 - i]
            else:
                signal3[i:i + 2 * length] = new
            i = i + 2 * length

    if nsignals > 3:
        signal4 = np.empty(shape=(10000))
        signal = 0
        for i in range(10000):
            if np.random.randint(1000) == 1:
                signal = signal + 0.1
            if np.random.randint(1000) == 2:
                signal = signal - 0.1
            signal4[i] = signal

        max4 = np.amax(signal4)
        min4 = np.amin(signal4)

        signal4 = 2 * (signal4 - min4) / (max4 - min4) - 1

    if nsignals > 4:
        signal5 = np.empty(shape=(10000))
        signal5[0] = 0
        for i in range(1, 10000):
            signal5[i] = signal5[i - 1] + np.random.randint(-5, 6) / 300

        max5 = np.amax(signal5)
        min5 = np.amin(signal5)

        signal5 = 2 * (signal5 - min5) / (max5 - min5) - 1

    if nsignals == 0:
        return None

    if nsignals == 1:
        return signal1

    if nsignals == 2:
        return np.column_stack((signal1, signal2))

    if nsignals == 3:
        return np.column_stack((signal1, signal2, signal3))

    if nsignals == 4:
        return np.column_stack((signal1, signal2, signal3, signal4))

    if nsignals == 5:
        return np.column_stack((signal1, signal2, signal3, signal4, signal5))

def generate_annotated_signals(iteration,nsignals):
    # Annotates each signal timestep with a higher-order feature type. 1 is plateau, 2 is peak, 3 is valley, 4 is rise,
    # 5 is decline, 6 is rough
    if nsignals > 5:
        print("Too many signals")
        return None
    if iteration > 2:
        print("Too many iterations")
        return None
    annotations = np.empty((10000,nsignals))
    if nsignals > 0:
        for i in range(0, 10000, 1000):
            annotations[i:i+500,0] = [2] * 500
            annotations[i+500:i+1000,0] = [3] * 500

        if iteration == 1:
            sine = list(range(10000))
            sine = [math.sin(math.pi * x / 500) for x in sine]
            sine[3209:3612] = [-0.3] * 403
            sine[6792:7150] = [-0.3] * 358
            sine[8523:8910] = [-0.3] * 387

            signal1 = np.asarray(sine)

            annotations[3209:3612,0] = [1] * 403
            annotations[6792:7150,0] = [1] * 358
            annotations[8523:8910,0] = [1] * 387

        if iteration == 2:
            sine = list(range(10000))
            sine = [math.sin(math.pi * x / 500) for x in sine]
            sine[1360:1733] = [-0.3] * 373
            sine[4306:4652] = [-0.3] * 346
            sine[8203:8619] = [-0.3] * 416
            sine[9699:10000] = [-0.3] * 301
            signal1 = np.asarray(sine)

            annotations[1360:1733, 0] = [1] * 373
            annotations[4306:4652, 0] = [1] * 346
            annotations[8203:8619, 0] = [1] * 416
            annotations[9699:10000, 0] = [1] * 301

    if nsignals > 1:
        if iteration == 1:
            signal2 = np.empty(shape=(10000))
            firstpart = list(range(2909))
            firstpart = [math.sin(math.pi * x / (2909)) for x in firstpart]
            signal2[0:2909] = firstpart
            annotations[0:2909,1] = [2] * 2909
            secondpart = list(range(600))
            secondpart = [math.sin(math.pi * x / 600) * -1 for x in secondpart]
            signal2[2909:3509] = secondpart
            annotations[2909:3509,1] = [3] * 600
            thirdpart = list(range(2983))
            thirdpart = [math.sin(math.pi * x / (2983)) for x in thirdpart]
            signal2[3509:6492] = thirdpart
            annotations[3509:6492,1] = [2] * 2983
            fourthpart = list(range(600))
            fourthpart = [math.sin(math.pi * x / 600) * -1 for x in fourthpart]
            signal2[6492:7092] = fourthpart
            annotations[6492:7092,1] = [3] * 600
            fifthpart = list(range(1131))
            fifthpart = [math.sin(math.pi * x / 1131) for x in fifthpart]
            signal2[7092:8223] = fifthpart
            annotations[7092:8223,1] = [2] * 1131
            sixthpart = list(range(600))
            sixthpart = [math.sin(math.pi * x / 600) * -1 for x in sixthpart]
            signal2[8223:8823] = sixthpart
            annotations[8223:8823,1] = [3] * 600
            seventhpart = list(range(1177))
            seventhpart = [math.sin(math.pi * x / 1177) for x in seventhpart]
            signal2[8823:10000] = seventhpart
            annotations[8823:10000,1] = [2] * 1177
        if iteration == 2:
            signal2 = np.empty(shape=(10000))
            firstpart = list(range(1060))
            firstpart = [math.sin(math.pi * x / (1060)) for x in firstpart]
            signal2[0:1060] = firstpart
            annotations[0:1060,1] = [2] * 1060
            secondpart = list(range(600))
            secondpart = [math.sin(math.pi * x / 600) * -1 for x in secondpart]
            signal2[1060:1660] = secondpart
            annotations[1060:1660,1] = [3] * 600
            thirdpart = list(range(2346))
            thirdpart = [math.sin(math.pi * x / (2346)) for x in thirdpart]
            signal2[1660:4006] = thirdpart
            annotations[1660:4006,1] = [2] * 2346
            fourthpart = list(range(600))
            fourthpart = [math.sin(math.pi * x / 600) * -1 for x in fourthpart]
            signal2[4006:4606] = fourthpart
            annotations[4006:4606,1] = [3] * 600
            fifthpart = list(range(3297))
            fifthpart = [math.sin(math.pi * x / 3297) for x in fifthpart]
            signal2[4606:7903] = fifthpart
            annotations[4606:7903,1] = [2] * 3297
            sixthpart = list(range(600))
            sixthpart = [math.sin(math.pi * x / 600) * -1 for x in sixthpart]
            signal2[7903:8503] = sixthpart
            annotations[7903:8503,1] = [3] * 600
            seventhpart = list(range(895))
            seventhpart = [math.sin(math.pi * x / 895) for x in seventhpart]
            signal2[8503:9398] = seventhpart
            annotations[8503:9398,1] = [2] * 895
            eighthpart = list(range(602))
            eighthpart = [math.sin(math.pi * x / 602) * -1 for x in eighthpart]
            signal2[9398:10000] = eighthpart
            annotations[9398:10000,1] = [3] * 602

    if nsignals > 2:
        signal3 = np.empty(shape=(10000))
        i = 0
        while i < 10000:
            direction = np.random.randint(2)
            length = np.random.randint(300, 600)
            new = list(range(2 * length))
            if direction == 1:
                new = [math.sin(math.pi * x / length - 0.5 * math.pi) * 0.5 + 0.5 for x in new]
                newann = [2] * 2 * length
            else:
                new = [math.sin(math.pi * x / length + 0.5 * math.pi) * 0.5 - 0.5 for x in new]
                newann = [3] * 2 * length
            if i + 2 * length > 10000:
                signal3[i:10000] = new[0:10000 - i]
                annotations[i:10000,2] = newann[0:10000-i]
            else:
                signal3[i:i + 2 * length] = new
                annotations[i:i + 2 * length,2] = newann
            i = i + 2 * length

    if nsignals > 3:
        signal4 = np.empty(shape=(10000))
        signal = 0
        for i in range(10000):
            annotations[i,3] = 1
            if np.random.randint(1000) == 1:
                signal = signal + 0.1
                annotations[i,3] = 4
            if np.random.randint(1000) == 2:
                signal = signal - 0.1
                annotations[i,3] = 5
            signal4[i] = signal

        max4 = np.amax(signal4)
        min4 = np.amin(signal4)

        signal4 = 2 * (signal4 - min4) / (max4 - min4) - 1



    if nsignals > 4:
        signal5 = np.empty(shape=(10000))
        signal5[0] = 0
        for i in range(1, 10000):
            signal5[i] = signal5[i - 1] + np.random.randint(-5, 6) / 300

        max5 = np.amax(signal5)
        min5 = np.amin(signal5)

        signal5 = 2 * (signal5 - min5) / (max5 - min5) - 1

        annotations[:,4] = [6] * 10000

    if nsignals == 0:
        return None

    if nsignals == 1:
        return signal1, annotations

    if nsignals == 2:
        return np.column_stack((signal1, signal2)), annotations

    if nsignals == 3:
        return np.column_stack((signal1, signal2, signal3)), annotations

    if nsignals == 4:
        return np.column_stack((signal1, signal2, signal3, signal4)), annotations

    if nsignals == 5:
        return np.column_stack((signal1, signal2, signal3, signal4, signal5)), annotations

def tokenize_annotations(annotations,nannotations):
    length = annotations.shape[0]
    signals = annotations.shape[1]
    tannotations = np.empty((length,signals,nannotations))
    a = list(range(nannotations))
    for i in range(length):
        for j in range(signals):
            tannotations[i][j] = a == annotations[i][j] - 1
    return tannotations

def visualise_signals(signals):
    plt.clf()
    fig, (ax) = plt.subplots(1,1,figsize=(10,3))
    ax.plot(list(range(10000)), signals[:,0], color='b')
    ax.plot(list(range(10000)), signals[:,1], color='tab:orange')
    ax.plot(list(range(10000)), signals[:,2], color='g')
    ax.plot(list(range(10000)), signals[:,3], color='r')
    ax.plot(list(range(10000)), signals[:,4], color='tab:purple')
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.7, box.height])
    ax.legend(('signal1','signal2', 'signal3', 'signal4', 'signal5'), loc='center left', bbox_to_anchor=(1, 0.5), fancybox=True)
    plt.savefig('signals.eps', format='eps')

def visualise_signal_segment(segment,timesteps):
    plt.clf()
    fig, (ax) = plt.subplots(1,1,figsize=(10,3))
    ax.plot(list(range(timesteps)), segment[:, 0], color='b')
    ax.plot(list(range(timesteps)), segment[:, 1], color='tab:orange')
    ax.plot(list(range(timesteps)), segment[:, 2], color='g')
    ax.plot(list(range(timesteps)), segment[:, 3], color='r')
    ax.plot(list(range(timesteps)), segment[:, 4], color='tab:purple')
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.7, box.height])
    ax.legend(('signal1', 'signal2', 'signal3', 'signal4', 'signal5'), loc='center left', bbox_to_anchor=(1, 0.5),
              fancybox=True)
    plt.show()

def visualise_pert(name,item_new,item_orig,timestep,timesteps,totalsteps,signals,othersignals,nsignals,signalnames,predictions,show_window):

    plt.clf()
    fig, (ax2) = plt.subplots(1, 1, figsize=(10, 3))
    

    colors = ['tab:orange', 'g', 'r', 'tab:purple', 'b', 'tab:pink', 'tab:cyan']
    colors2 = [mcd.CSS4_COLORS['moccasin'],mcd.CSS4_COLORS['lightgreen'],mcd.CSS4_COLORS['lightcoral'],mcd.CSS4_COLORS['thistle'],mcd.CSS4_COLORS['lightskyblue'],mcd.CSS4_COLORS['lavenderblush'],mcd.CSS4_COLORS['lightcyan']]

    ax2.set_xlabel('timesteps')
    ax2.set_ylabel('signals')
    lines = []
    linespert = []
    for i in range(nsignals):
        lines.append(ax2.plot(list(range(timestep-timesteps,timestep)), item_orig[:,i],color = colors[i],label=signalnames[i]))
        linespert.append(ax2.plot(list(range(timestep-timesteps,timestep)), item_new[:, i],color = colors2[i],label='pert'))

    lines = lines+linespert
    lines = [item for sublist in lines for item in sublist]

    box = ax2.get_position()
    ax2.set_position([box.x0, box.y0, box.width * 0.6, box.height])

    ax2.legend(handles= lines,loc='center left', bbox_to_anchor=(1, 0.5),ncol=2, fancybox=True)


    plt.savefig('perturbation{}'.format(name))
    
def visualise_LIME(item,analysis,nsignals,signals,signalnames,predictions,timestep,timesteps,totalsteps):
    plt.clf()
    fig, (ax2, ax3) = plt.subplots(2, 1, figsize=(10, 6))

    colors = ['tab:orange', 'g', 'r', 'tab:purple', 'b', 'tab:pink', 'tab:cyan']
  
    ax2.set_xlabel('timesteps')
    ax2.set_ylabel('signals')
    lines = []
    for i in range(nsignals):
        lines.append(ax2.plot(list(range(timestep-timesteps,timestep)), item[:,i],color = colors[i],label=signalnames[i]))

    lines = [item for sublist in lines for item in sublist]

    box = ax2.get_position()
    ax2.set_position([box.x0, box.y0, box.width * 0.6, box.height])

    ax2.legend(handles= lines,loc='center left', bbox_to_anchor=(1, 0.5), fancybox=True)
    
    ax3.set_xlabel('timesteps')
    ax3.set_ylabel('relevance')
    lines=[]
    for i in range(nsignals):
        lines.append(ax3.plot(list(range(timestep-timesteps,timestep)), analysis[:,i],color = colors[i],label=signalnames[i]))
        
    lines = [item for sublist in lines for item in sublist]

    box = ax3.get_position()
    ax3.set_position([box.x0, box.y0, box.width * 0.6, box.height])

    ax3.legend(handles= lines,loc='center left', bbox_to_anchor=(1, 0.5), fancybox=True)
    
    plt.savefig('LIME explanation')

def visualise_AM(signals,node,timesteps,nsignals,signalnames):
    plt.clf()
    fig, (ax) = plt.subplots(1, 1, figsize=(10, 3))
    colors = ['tab:orange', 'g', 'r', 'tab:purple', 'b', 'tab:pink', 'tab:cyan']
    for i in range(nsignals):
        ax.plot(list(range(timesteps)),signals[:,i], color=colors[i])

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.6, box.height])
    ax.legend(('signal2','signal3','signal4','signal5'),loc='center left', bbox_to_anchor=(1, 0.5),fancybox=True)
    ax.legend(signalnames)
    plt.savefig('AM{}'.format(node))

def visualise_analysis(analysis,timestep,totalsteps,type,signals,signal_names,show_window,predictions):
    steps = analysis.shape[0]
    nsignals = signals.shape[1]

    plt.clf()
    fig, (ax1, ax2, ax3) = plt.subplots(3,1,figsize=(6,10))

    ax1a = ax1.twinx()
    ax1a.set_xlabel('timesteps')
    ax1a.set_ylabel('anomaly probability', color='y')
    ax1a.plot(list(range(timestep-steps-show_window,timestep+show_window)), predictions[timestep-steps-show_window:timestep+show_window], color='y')
    ax1a.tick_params(axis='y', labelcolor='y')

    ax1.set_ylabel('signals')
    ax1.set_xlabel('timesteps')
    colors = ['tab:orange','g','r','tab:purple','b','tab:pink','tab:cyan']
    for i in range(nsignals):
        ax1.plot(list(range(timestep-steps-show_window,timestep+show_window)), signals[timestep-steps-show_window:timestep+show_window:,i], color=colors[i])
    ax1.axvspan(timestep-steps, timestep, color='red', alpha=0.5)

    ax2.set_xlabel('timesteps')
    ax2.set_ylabel('signals')
    for i in range(nsignals):
        ax2.plot(list(range(timestep - steps, timestep)), signals[timestep-steps:timestep,i], color=colors[i])

    for i in range(nsignals):
        ax3.plot(list(range(timestep - steps, timestep)), analysis[:, i], color=colors[i])
    ax3.set_xlabel('timesteps')
    ax3.set_ylabel('relevance score')

    ax3.legend(signal_names)

    con = ConnectionPatch(xyA=(timestep-steps,np.amin(predictions)), xyB=(timestep-steps,np.amax(signals[timestep-steps:timestep])),
                          coordsA="data", coordsB="data",
                          axesA=ax1a, axesB=ax2)

    ax1a.add_artist(con)

    con = ConnectionPatch(xyA=(timestep, np.amin(predictions)), xyB=(timestep, np.amax(signals[timestep-steps:timestep])),
                          coordsA="data", coordsB="data",
                          axesA=ax1a, axesB=ax2)

    ax1a.add_artist(con)

    plt.savefig('{}.png'.format(type))

