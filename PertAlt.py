import tensorflow as tf

from keras.layers import Dense, Flatten
from keras import Input, Model
import keras
import keras.models
import keras.layers
from keras.models import model_from_json

import numpy as np
import pandas as pd

import scipy

from tcn import TCN

import innvestigate
from innvestigate import utils as iutils
from innvestigate.utils.keras import checks as kchecks
from innvestigate.utils.keras.graph import copy_layer_wo_activation

import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch
import matplotlib._color_data as mcd
import math

from functions import generate_signals, visualise_pert
from illustrationfunctions import visualise_analysis, visualise_shapelet, visualise_difference_segments, visualise_relevance_selection, visualise_LRP_selection
#from segmfunctions import multiple_segment_smoothing, untokenize_annotations

class difference_segment:
    def __init__(self,signal,start):
        self.type = 'none'
        self.signal = signal
        self.start = start
        self.end = -1
        self.origaverage = 0
        self.protaverage = 0
        self.tippingaverage = 0
        self.origtype = 0
        self.prottype = 0
        self.item = 0

class shapelet_candidate:
    def __init__(self,signal,start,end):
        self.signal = signal
        self.start = start
        self.end = end

class shapelet:
    def __init__(self,signal,length, shape):
        self.signal = signal
        self.length = length
        self.shape = shape
        self.item = 0

# load json and create model
json_file = open('modelsmall0.35.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
# load weights into new model
model.load_weights("smallmodel0.35.h5")

for i, layer in enumerate(model.layers):
    layer._name = 'layer_' + str(i)

layer = model.layers[-1]
layer_wo_act = copy_layer_wo_activation(layer)
output_layer = layer_wo_act(layer.input)
model_wo_sigm = keras.models.Model(inputs=model.inputs,outputs=[output_layer])

# serialize model to JSON
model_json = model_wo_sigm.to_json()
with open("model_wo_sigm.json", "w") as json_file:
    json_file.write(model_json)

model_wo_sigm.save_weights("model_wo_sigm.h5")

timesteps2 = 256
nsignals = 29

if __name__ == "__main__":
    test_x = np.load('test_x.npy')

    predictions = model.predict(test_x)

    maxs = np.load('maxs.npy')
    mins = np.load('mins.npy')

# The 'worst' neighbour is found by performing gradient descent or ascent on all timesteps of all features.
def neighbour(item,item_orig,model,timesteps2,nsignals,maxs,mins,type):
    # Parameters:
    # item: the item at the current perturbation point
    # item_orig: the original item
    # mode: the prediction model to be explained
    # timesteps2: the number of timesteps in an item
    # nsignals: the number of signals in an item
    # maxs: the maximum values that each signal may take
    # mins: the minimum values that each signal may take
    # type : 0 means gradient descent, 1 means ascent

    neighbour = item.copy()

    prob = model.predict(item.reshape(1,timesteps2,nsignals))
    prob = tf.constant(prob[0])

    # Compute the gradients.
    item = item.astype(np.float32)
    item = item.reshape(1, timesteps2, nsignals)
    item = tf.constant(item)

    item_orig = item_orig.astype(np.float32)
    item_orig = item_orig.reshape(1, timesteps2, nsignals)
    item_orig = tf.constant(item_orig)

    pred = model_wo_sigm(item)

    roughness = tf.constant(0,dtype=tf.float32)
    for i in range(nsignals):
        roughness = roughness + tf.math.reduce_std(item[:, 1:, i] - item[:, :-1, i])
    roughness = roughness / nsignals

    difference = tf.math.reduce_mean(2 * (tf.math.exp(tf.math.abs(item-item_orig))-1))

    if type == 0:
        grads = tf.gradients((prob + 0.55) * pred[0][0] + 500* roughness + 500*difference, item)
    else:
        grads = tf.gradients((1 - prob + 0.55) * pred[0][0] - 500 * roughness - 500 * difference, item)
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        model.load_weights("smallmodel0.35.h5")
        model_wo_sigm.load_weights("model_wo_sigm.h5")
        grads = sess.run(grads[0][0])

    # Perform gradient descent or ascent, remove impossible values.
    for i in range(timesteps2):
        for j in range(nsignals):
            #neighbour[i][j] = neighbour[i][j] - grads[i][j] + ((np.random.random_sample() * 2 - 1) / 100 )
            if type == 0:
                neighbour[i][j] = neighbour[i][j] - 0.01 * grads[i][j]
            else:
                neighbour[i][j] = neighbour[i][j] + 0.01 * grads[i][j]
            if neighbour[i][j] > maxs[j]:
                neighbour[i][j] = maxs[j]
            if neighbour[i][j] < mins[j]:
                neighbour[i][j] = mins[j]

    return neighbour

# This function performs either gradient descent (if type = 0) or gradient ascent (if type = 1) to arrive at a perturbed signal
# for the original signal so that the prediction is flipped to negative (type = 0) or positive (type = 1).
def perturbation(item_orig,model,iterations,timesteps2,nsignals,maxs,mins,type):
    item = item_orig
    tipping_point = None
    counter = 0
    check = 0

    # For each of the iterations, perform gradients descent / ascent
    for i in range(iterations):
        print("Iteration {}".format(i))
        for i in range(29):
            if np.std(item[:,i]) < 0.01:
                item[:,i] = item[:,i] + (np.random.rand(timesteps2) - 0.5) / 100
        # The neighbour is found by adding or subtracting the gradients.
        nb = neighbour(item.copy(),item_orig.copy(), model,timesteps2,nsignals,maxs,mins,type)

        # Calculate several scores for bookkeeping.
        item_score = model.predict(item.reshape((1, timesteps2, nsignals)))[0]
        neighbour_score = model.predict(nb.reshape((1, timesteps2, nsignals)))[0]
        print(item_score)
        print(neighbour_score)
        item_roughness = 0
        for i in range(nsignals):
            item_roughness = item_roughness + np.std(np.diff(item[:, i]))
        item_roughness = item_roughness / nsignals
        neighbour_roughness = 0
        for i in range(nsignals):
            neighbour_roughness = neighbour_roughness + np.std(np.diff(nb[:, i]))
        neighbour_roughness = neighbour_roughness / nsignals
        print(item_roughness)
        print(neighbour_roughness)
        item_diff = np.mean(np.abs(item-item_orig))
        neighbour_diff = np.mean(np.abs(nb-item_orig))
        print(item_diff)
        print(neighbour_diff)

        # Case: we are doing gradient descent to lower the prediction score
        if type == 0:
            # If the prediction score or one of the penalty terms is lower, continue
            if neighbour_score < item_score or neighbour_roughness < item_roughness or neighbour_diff < item_diff:
                item = nb.copy()
                if neighbour_score > 0.1:
                    counter = 0
            # If not, increase the counter and stop running if the counter has gone on too much
            else:
                item = nb.copy()
                counter = counter + 1
            # If we cross 0.5 we add the current neighbour as the tipping point
            if neighbour_score < 0.5 and check == 0:
                tipping_point = nb.copy()
                check = 1
            # If we have reached a very low neighbour score we also increase the counter and stop running if the counter
            # is too high
            if neighbour_score < 0.1:
                counter = counter+1
            if neighbour_score < 0.1 and counter > 5:
                return item, tipping_point
        # Case: we are doing gradient ascent to make the prediction score higher
        if type == 1:
            if neighbour_score > item_score or neighbour_roughness < item_roughness or neighbour_diff < item_diff:
                item = nb.copy()
                if neighbour_score < 0.9:
                    counter = 0
            else:
                item = nb.copy()
                counter = counter + 1
            if neighbour_score > 0.5 and check == 0:
                tipping_point = nb.copy()
                check = 1
            if neighbour_score > 0.9:
                counter = counter+1
            if neighbour_score > 0.9 and counter > 5:
                return item, tipping_point
    return item, tipping_point

if __name__ == "__main__":
# This file can be run either on its own or be called by ruleset_generation. If
# it runs on its own, the following code is executed to analyse one specific
# item.
    
    indices1 = [427,5285,19182,19531,20318,20701]
    indices2 = [439,5298,19195,19546,20331,20714]

    timestep = 19540

    # This file contains the downsampled data.
    data = pd.read_csv('reduced_data')
    
    # The first column is an artifact; remove it.
    data = data.iloc[:,1:33]

    # Extract all signals and signal names that matter. The setpoint and rotation
    # speed values are kept apart.
    signalnames = data.columns.to_numpy()
    signalnames = signalnames[0:30]
    signalnames = np.delete(signalnames, 10)
    signals = data.to_numpy()
    differentsignals = signals[:,np.array([10,30])]
    signals = signals[:, 0:30]
    signals = np.delete(signals, 10, 1)

    predictions = model.predict(test_x)

    total_predictions = np.zeros(data.shape[0])

    # Iterate over each timestep. If it is part of the test set it is given
    # the corresponding value; otherwise, its prediction is set to 0 by default.
    i = 0
    j = 427
    k = 0
    for i in range(len(predictions)):
        total_predictions[j] = predictions[i]
        j = j + 1
        if j == indices2[k] and k < 5:
            k = k + 1
            j = indices1[k]

    # Select the index in the test set that corresponds to the item of interest.
    index1 = timestep
    index2 = 0
    help = 0
    switch = 0
    auxiliary_index = indices1[help]
    while auxiliary_index != index1:
        if auxiliary_index == indices2[help]:
            help = help + 1
            auxiliary_index = indices1[help]
        else:
            auxiliary_index = auxiliary_index + 1
        index2 = index2 + 1

    item_orig = test_x[index2]

    for i in range(29):
        if np.std(item_orig[:,i]) < 0.01:
            item_orig[:,i] = item_orig[:,i] + (np.random.rand(timesteps2) - 0.5) / 100

    
    predict = model.predict(item_orig.reshape(1,timesteps2,29))[0]
    print(predict)

    if predict > 0.5:
        kind = 0
    else:
        kind = 1
        
    # Load the statistics for all signals. Max and min values for each signal are 
    # normalised using Z normalisation.
    maxs = np.load('maxs.npy')
    mins = np.load('mins.npy')
    means = np.load('means.npy')
    stds = np.load('stds.npy')
    for sign in range(nsignals):
        maxs[sign] = (maxs[sign] - means[sign]) / stds[sign]
        mins[sign] = (mins[sign] - means[sign]) / stds[sign]

    prototype, tipping_point = perturbation(item_orig,model,25,timesteps2,nsignals,maxs,mins,kind)
    
    # Find the most interesting signals
    differences = prototype - item_orig
    differencesabs = np.abs(differences)
    totaldiff = np.zeros(29)
    for i in range(29):
        totaldiff[i] = np.sum(differencesabs[:, i])
    print(totaldiff)
    ranking = np.argsort(-totaldiff)
    print(totaldiff[ranking])
    
    # Normalise the signals according to Z-score.
    for i in range(29):
        if stds[i] != 0:
            signals[:,i] = (signals[:,i] - means[i]) / stds[i]
        else:
            signals[:,i] = np.zeros(len(total_predictions))


    selection = ranking[0:4]
    item_orig2 = item_orig[:, selection]
    prototype2 = prototype[:, selection]
    signals2 = signals[:, selection]
    signalnames2 = signalnames[selection]
    differencesabs2 = differencesabs[:,selection]
    visualise_relevance_selection(item_orig2,prototype2,signalnames2,differencesabs2,timesteps2)
    visualise_pert('negative', prototype2, item_orig2, index1, timesteps2, len(total_predictions), signals2,differentsignals, 4, signalnames2,
                       total_predictions, 3000)
    #visualise_pert(prototype,item_orig,timestep,timesteps2,signals,model.predict(test_x),timestep,5)
    print("image ready")

def find_value_segments(item_orig,prototype,timesteps2,nsignals,minlength,ranking):

    difference = prototype - item_orig

    differenceflat = np.reshape(difference,newshape=nsignals * timesteps2,order='F')
    differenceabs = np.abs(differenceflat)
    threshold = 2 * np.std(differenceabs)
    filter = differenceabs > threshold
    print("simplicity score : {}".format(np.sum(filter)))
    simplicity_score_2 = 0
    for j in range(nsignals):
        if np.sum(filter[j*timesteps2:j*timesteps2+timesteps2]) > 0:
            simplicity_score_2 = simplicity_score_2 + 1
    print("simplicity score 2 : {}".format(simplicity_score_2))
    segments = []
    for i in range(nsignals):
        switch = 0
        for j in range(timesteps2):
            if switch == 0:
                if filter[i*timesteps2 + j]:
                    switch = 1
                    segment = difference_segment(signal=ranking[i],start=j)
            else:
                if not(filter[i*timesteps2 + j]):
                    switch = 0
                    segment.end = j
                    if segment.end - segment.start >= minlength:
                        segments.append(segment)
                if j == timesteps2 - 1:
                    segment.end = j+1
                    if segment.end - segment.start >= minlength:
                        segments.append(segment)

    total_segments = 0
    for segment in segments:
        total_segments = total_segments +1
        segment.origaverage = np.mean(item_orig[segment.start:segment.end,np.where(ranking==segment.signal)[0][0]])
        segment.protaverage = np.mean(prototype[segment.start:segment.end,np.where(ranking==segment.signal)[0][0]])
        if segment.origaverage > segment.protaverage:
            segment.type = 'higher'
        else:
            segment.type = 'lower'
        #segment.tippingaverage = np.mean(tipping_point[segment.start:segment.end,segment.signal])
        #segment.origtype = int(scipy.stats.mode(annotations1[segment.start:segment.end,segment.signal])[0][0])
        #segment.prottype = int(scipy.stats.mode(annotations2[segment.start:segment.end,segment.signal])[0][0])

    print("Number of value segments : {}".format(total_segments))
    return segments, threshold

def find_shapelets(item,relevant_nodes,int_model,timesteps2,nsignals,maxlength):

    analyzer = innvestigate.create_analyzer("lrp.epsilon", int_model, allow_lambda_layers=True,
                                            neuron_selection_mode="index")

    shapelet_candidates = []
    shapelets_final = []
    
    for node in relevant_nodes:
        analysis = analyzer.analyze(item.reshape(1,timesteps2,nsignals),node)[0]
        analysis = np.abs(analysis)

        maxrelevance = 0
        bestsignal = 0
        for i in range(nsignals):
            if(np.sum(np.abs(analysis[:,i])) > maxrelevance):
                maxrelevance = np.sum(np.abs(analysis[:,i]))
                bestsignal = i
        analysis = analysis[:,bestsignal]

        #visualise_analysis(analysis, timestep, 'LRP {}'.format(node), signal6, timesteps2, predictions, 1)

        threshold = np.std(analysis) + np.mean(analysis)
        
        #visualise_LRP_selection(analysis,anstd,timesteps2,item,signalnames,bestsignal,z)
        
        relevant_steps = np.argwhere(analysis > threshold)[:,0]

        best_size = (3/4) * len(relevant_steps)
        check = 0

        for beginindex,beginstep in enumerate(relevant_steps):
            if beginstep + maxlength >= timesteps2:
                endstep = timesteps2 - 1
            else:
                endstep = beginstep + maxlength
            endindex = np.searchsorted(relevant_steps,endstep)
            endstep = relevant_steps[endindex-1]
            if len(relevant_steps[beginindex:endindex]) >= best_size:
                check = 1
                shapcand = shapelet_candidate(bestsignal,beginstep,endstep)
                best_size = len(relevant_steps[beginindex:endindex])

        if check == 1:
            shapelet_candidates.append(shapcand)
            #print(bestsignal)
            #visualise_shapelet(signals[:,bestsignal+1], shapcand, timestep, 256, timesteps2, node, analysis)

    # Next, some of the shapelets will be similar or identical. These shapelets are combined.
    check = 1
    while(len(shapelet_candidates) > 1 and check==1):
        for cand1 in shapelet_candidates:
            check = 0
            for cand2 in shapelet_candidates:
                if cand1 != cand2:
                    if cand1.signal == cand2.signal and np.abs(cand1.start-cand2.start) < maxlength / 5 and np.abs(cand1.end - cand2.end) < maxlength / 5:
                        combined_cand = shapelet_candidate(cand1.signal,np.min([cand1.start,cand2.start]),np.max([cand1.end,cand2.end]))
                        shapelet_candidates.remove(cand1)
                        shapelet_candidates.remove(cand2)
                        shapelet_candidates.append(combined_cand)
                        check = 1
                        break
            if check == 1:
                break

    for sl in shapelet_candidates:
        shapelet_final = shapelet(sl.signal,sl.end-sl.start,item[sl.start:sl.end,sl.signal])
        shapelets_final.append(shapelet_final)

    return shapelets_final

if __name__ == "__main__":

    valuesegments, threshold = find_value_segments(item_orig2,prototype2,timesteps2,4,2,ranking)
    
    valuesegments2 = []
    
    for segment in valuesegments:
        if segment.signal in selection:
            print("Attention for signal {}".format(segment.signal))
            valuesegments2.append(segment)
    
    visualise_difference_segments(item_orig2, prototype2, timestep, 256, valuesegments2, signalnames2,4, ranking)

    int_model = Model(inputs=model.inputs, outputs=[model.layers[-2].output])

    for i, layer in enumerate(int_model.layers):
        layer._name = 'layer_' + str(i)

    item_scores = int_model.predict(item_orig.reshape(1, timesteps2, nsignals))[0]
    item_ranking = np.argsort(-item_scores)
    relevant_nodes_item = item_ranking[:4]
    print(relevant_nodes_item)

    prototype_scores = int_model.predict(prototype.reshape(1, timesteps2, nsignals))[0]
    prototype_ranking = np.argsort(-prototype_scores)
    relevant_nodes_prototype = prototype_ranking[:4]

    for node in relevant_nodes_item:
        if node in relevant_nodes_prototype:
            relevant_nodes_item = np.delete(relevant_nodes_item,np.argwhere(relevant_nodes_item==node))
            relevant_nodes_prototype = np.delete(relevant_nodes_prototype,np.argwhere(relevant_nodes_prototype==node))

    print(relevant_nodes_item)

    shapelets_item = find_shapelets(item_orig,relevant_nodes_item,int_model,timesteps2,nsignals,100)
    shapelets_prototype = find_shapelets(prototype,relevant_nodes_prototype,int_model,timesteps2,nsignals,100)

    plt.clf()
    fig, (ax1) = plt.subplots(1,1,figsize=(10,3))
    lines = []
    for slx in shapelets_item:
        lines.append(ax1.plot(range(slx.length),slx.shape,label = signalnames[slx.signal]))
    lines = [item for sublist in lines for item in sublist]
    plt.legend(handles=lines)
    plt.savefig("combined_shapelet")

    print(len(valuesegments))
    print(len(shapelets_item))

    print("The prediction is explained by",end=" ")
    i = len(valuesegments)
    for segment in valuesegments:
        print("the segment of signal {} between {} and {}".format(segment.signal,segment.start,segment.end),end=" ")
        #if segment.origtype != segment.prottype:
            #print("being {} rather than {}".format(types[segment.origtype],types[segment.prottype]),end=" ")
            #if np.abs(segment.origaverage - segment.protaverage) > threshold:
                #print("and",end=" ")
        if np.abs(segment.origaverage-segment.protaverage) > threshold:
            if segment.origaverage-segment.protaverage > 0:
                print("being on average higher than {}".format(segment.tippingaverage),end=" ")
            else:
                print("being on average lower than {}".format(segment.tippingaverage), end=" ")
        if i > 1 or len(shapelets_item)>0:
            print("and",end=" ")
        else:
            print(".")
        i = i-1

    i = len(shapelets_item)
    for shapelet in shapelets_item:
        print("signal {} having the following shapelet: {}".format(shapelet.signal,shapelet.shape),end="")
        if i > 1:
            print("and",end=" ")
        else:
            print(".")
        i = i-1
