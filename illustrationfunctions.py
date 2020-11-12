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

def visualise_analysis(analysis,timestep,type,signals,timesteps2,predictions,nsignals):
    steps = analysis.shape[0]

    signal6 = signals
    if nsignals > 1:
        signal7 = signals[:, 2]
        signal8 = signals[:, 3]
        signal9 = signals[:, 4]

    plt.clf()
    fig, (ax1, ax2, ax3) = plt.subplots(3,1,figsize=(6,10))

    ax1a = ax1.twinx()
    ax1a.set_xlabel('timesteps')
    ax1a.set_ylabel('anomaly probability', color='y')
    ax1a.plot(list(range(10000 - timesteps2)), predictions, color='y')
    ax1a.tick_params(axis='y', labelcolor='y')

    ax1.set_ylabel('signals')
    ax1.set_xlabel('timesteps')
    ax1.plot(list(range(10000 - timesteps2)), signal6[timesteps2:10000], color='tab:orange')
    if nsignals > 1:
        ax1.plot(list(range(10000 - timesteps2)), signal7[timesteps2:10000], color='g')
        ax1.plot(list(range(10000 - timesteps2)), signal8[timesteps2:10000], color='r')
        ax1.plot(list(range(10000 - timesteps2)), signal9[timesteps2:10000], color='tab:purple')
    ax1.axvspan(timestep-steps, timestep, color='red', alpha=0.5)

    ax2.set_xlabel('timesteps')
    ax2.set_ylabel('signals')
    ax2.plot(list(range(timestep - steps, timestep)), signal6[timestep-steps+timesteps2:timestep+timesteps2], color='tab:orange')
    if nsignals > 1:
        ax2.plot(list(range(timestep - steps, timestep)), signal7[timestep-steps+timesteps2:timestep+timesteps2], color='g')
        ax2.plot(list(range(timestep - steps, timestep)), signal8[timestep-steps+timesteps2:timestep+timesteps2], color='r')
        ax2.plot(list(range(timestep - steps, timestep)), signal9[timestep-steps+timesteps2:timestep+timesteps2], color='tab:purple')

    ax3.plot(list(range(timestep - steps, timestep)), analysis, color='tab:orange')
    if nsignals > 1:
        ax3.plot(list(range(timestep - steps, timestep)), analysis[:, 1], color='g')
        ax3.plot(list(range(timestep - steps, timestep)), analysis[:, 2], color='r')
        ax3.plot(list(range(timestep - steps, timestep)), analysis[:, 3], color='tab:purple')
    ax3.set_xlabel('timesteps')
    ax3.set_ylabel('relevance score')
    std = np.std(analysis)
    ax3.plot(list(range(timestep - steps,timestep)),[std] * steps, color = 'g')

    if nsignals == 1:
        ax3.legend(('signal 2','threshold'))
    else:
        ax3.legend(('signal 2', 'signal 3', 'signal 4', 'signal 5'))

    con = ConnectionPatch(xyA=(timestep-steps,np.amin(predictions)), xyB=(timestep-steps,np.amax(signals[timestep-steps+timesteps2:timestep+timesteps2])),
                          coordsA="data", coordsB="data",
                          axesA=ax1a, axesB=ax2)

    ax1a.add_artist(con)

    con = ConnectionPatch(xyA=(timestep, np.amin(predictions)), xyB=(timestep, np.amax(signals[timestep-steps+timesteps2:timestep+timesteps2])),
                          coordsA="data", coordsB="data",
                          axesA=ax1a, axesB=ax2)

    ax1a.add_artist(con)

    plt.savefig('{}.png'.format(type))

def visualise_shapelet(signal,candidate_shapelet,timestep,steps,timesteps2,node,analysis):
    plt.clf()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 6))
    ax1.plot(list(range(timestep-steps,timestep)),signal[timestep - steps + timesteps2:timestep + timesteps2],color='tab:orange')
    ax1.plot(list(range(timestep-steps+candidate_shapelet.start,timestep-steps+candidate_shapelet.end)),signal[timestep-steps+timesteps2+candidate_shapelet.start:timestep-steps+timesteps2+candidate_shapelet.end],color='red')
    ax1.legend(('signal 1','identified shapelet'))
    ax2.plot(list(range(timestep - steps, timestep)), analysis, color='tab:orange')
    std = np.std(analysis)
    ax2.plot(list(range(timestep - steps, timestep)), [std] * steps, color='g')
    ax2.legend(('LRP scores','threshold'))
    plt.savefig('candidate_shapelet{}'.format(node))

def visualise_difference_segments(item_orig,item_new,timestep,steps,diffsegments,signalnames,nsignals,ranking):
    plt.clf()

    fig, (ax2) = plt.subplots(1, 1, figsize=(10, 3))

    ax2.set_xlabel('timesteps')
    ax2.set_ylabel('signals')

    ax2.plot(list(range(timestep - steps, timestep)), item_orig[:, 0], color='b')
    ax2.plot(list(range(timestep - steps, timestep)), item_orig[:, 1], color='r')
    ax2.plot(list(range(timestep - steps, timestep)), item_orig[:, 2], color='tab:orange')
    ax2.plot(list(range(timestep - steps, timestep)), item_orig[:, 3], color='tab:purple')
    if nsignals > 4:
        ax2.plot(list(range(timestep - steps, timestep)), item_orig[:, 4], color='g')
    ax2.plot(list(range(timestep - steps, timestep)), item_new[:, 0], color=mcd.CSS4_COLORS['cyan'])
    ax2.plot(list(range(timestep - steps, timestep)), item_new[:, 1], color=mcd.CSS4_COLORS['lightcoral'])
    ax2.plot(list(range(timestep - steps, timestep)), item_new[:, 2], color=mcd.CSS4_COLORS['moccasin'])
    ax2.plot(list(range(timestep - steps, timestep)), item_new[:, 3], color=mcd.CSS4_COLORS['thistle'])
    if nsignals > 4:
        ax2.plot(list(range(timestep - steps, timestep)), item_new[:, 4], color=mcd.CSS4_COLORS['lightgreen'])

    box = ax2.get_position()
    ax2.set_position([box.x0, box.y0, box.width * 0.6, box.height])
    
    if nsignals > 4:
        ax2.legend(
            (signalnames[0], signalnames[1], signalnames[2], signalnames[3], signalnames[4], 'pert.', 'pert.', 'pert.', 'pert.', 'pert.'),
            loc='center left', bbox_to_anchor=(1, 0.5), ncol=2, fancybox=True)
    
    else:
         ax2.legend(
            (signalnames[0], signalnames[1], signalnames[2], signalnames[3], 'pert.', 'pert.', 'pert.', 'pert.'),
            loc='center left', bbox_to_anchor=(1, 0.5), ncol=2, fancybox=True)

    for segment in diffsegments:
        plt.fill_between(list(range(segment.start+timestep-steps,segment.end+timestep-steps)), item_orig[segment.start:segment.end,np.where(ranking==segment.signal)[0][0]], item_new[segment.start:segment.end,np.where(ranking==segment.signal)[0][0]],color=mcd.CSS4_COLORS['lightpink'])
    plt.savefig('value segments')
    
def visualise_relevance_selection(item_orig2,prototype2,signalnames2,differencesabs2,timesteps2):
    plt.clf()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 9))

    colors = ['tab:orange', 'g', 'r', 'tab:purple', 'b', 'tab:pink', 'tab:cyan']
    colors2 = [mcd.CSS4_COLORS['moccasin'], mcd.CSS4_COLORS['lightgreen'], mcd.CSS4_COLORS['lightcoral'],
           mcd.CSS4_COLORS['thistle'], mcd.CSS4_COLORS['lightskyblue'], mcd.CSS4_COLORS['lavenderblush'],
           mcd.CSS4_COLORS['lightcyan']]

    ax1.set_xlabel('timesteps')
    ax1.set_ylabel('signals')
    lines = []
    linespert = []
    for i in range(4):
        lines.append(ax1.plot(list(range(timesteps2)), item_orig2[:,i],color = colors[i],label=signalnames2[i]))
        linespert.append(ax1.plot(list(range(timesteps2)), prototype2[:, i],color = colors2[i],label='pert'))
    lines = lines + linespert
    lines = [item for sublist in lines for item in sublist]

    box = ax1.get_position()
    ax1.set_position([box.x0, box.y0, box.width * 0.6, box.height])

    ax1.legend(handles=lines, loc='center left', bbox_to_anchor=(1, 0.5), ncol=2, fancybox=True)

    for i in range(4):
        ax2.plot(list(range(timesteps2)),differencesabs2[:,i],color=colors[i])

    ax2.set_xlabel('timesteps')
    ax2.set_ylabel('difference score')

    dastd = np.std(differencesabs2)
    print(dastd)
    ax2.plot(list(range(timesteps2)),[2 * dastd] * timesteps2)

    box = ax2.get_position()
    ax2.set_position([box.x0, box.y0, box.width * 0.6, box.height])

    signalnames2 = np.append(signalnames2,'2 * STD threshold')
    ax2.legend(signalnames2,loc='center left',bbox_to_anchor=(1,0.5),fancybox=True)

    fig.savefig('relevant timestep selection')

def visualise_LRP_selection(analysis,anstd,timesteps2,item,signalnames,bestsignal,z):
    plt.clf()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))
    ax1.plot(list(range(timesteps2)),item[:,bestsignal])
    ax1.set_xlabel('timesteps')
    ax1.set_ylabel('signal score')
    ax1.legend([signalnames[bestsignal]])
    ax2.plot(list(range(timesteps2)),analysis)
    ax2.plot(list(range(timesteps2)),[anstd]*timesteps2)
    ax2.set_xlabel('timesteps')
    ax2.set_ylabel('LRP score')
    ax2.legend(['LRP score','1 STD threshold'])
    fig.savefig('LRP relevance selection {}'.format(z))
    
def visualise_condition(item,index,signal,signalname,ntimesteps,type,begin,end,value):
    plt.clf()
    fig, (ax1) = plt.subplots(1, 1, figsize=(10,4))
    ax1.plot(list(range(-ntimesteps,0)),item[:,signal],color='b')
    ax1.plot(list(range(-(ntimesteps - begin),-(ntimesteps - end))),[value] * (end-begin),color = 'r')
    ax1.legend([signalname,'threshold value'])
    fig.savefig('condition_example{}'.format(index))