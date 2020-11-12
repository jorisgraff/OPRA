# Generate rulesets that capture the model behaviour. First, we select singular candidate rules by combining overlapping
# HOAs. Then, we generate composite generate rules by means of apriori. Finally, we use a local search algorithm to combine
# all the candidate rules.

from keras.models import model_from_json, Model
import keras.utils

import numpy as np
import pandas as pd

from tcn import TCN

import math

import random

from innvestigate import utils as iutils
from innvestigate.utils.keras import checks as kchecks
from innvestigate.utils.keras.graph import copy_layer_wo_activation

from apyori import apriori

from PertAlt import perturbation, find_value_segments, find_shapelets
from functions import visualise_pert

import matplotlib.pyplot as plt

class value_condition:
    def __init__(self,type,signal,start):
        self.type = type
        self.signal = signal
        self.start = start
        self.end = -1
        self.origaverage = 0
        self.protaverage = 0
        self.tippingaverage = 0
        self.items = []

class shapelet_condition:
    def __init__(self,signal,length,shape,type):
        self.signal = signal
        self.length = length
        self.shape = shape
        self.items = []
        self.type = type

# Define constants

nsignals = 29
ntimesteps = 256

nitems = 30

support = 0.15 * nitems

segments = []
segmentshigher = []
segmentslower = []
shapelets_orig = []
shapelets_new = []

# load json and create model
json_file = open('modelsmall0.35.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
# load weights into new model
model.load_weights("smallmodel0.35.h5")

# Create a model without sigmoid for gradient a/descent.
layer = model.layers[-1]
layer_wo_act = copy_layer_wo_activation(layer)
output_layer = layer_wo_act(layer.input)
model_wo_sigm = keras.models.Model(inputs=model.inputs,outputs=[output_layer])

model_wo_sigm.save_weights("model_wo_sigm.h5")

# Create an intermediate model for shapelet detection.

int_model = Model(input=model.inputs, output=[model.layers[-2].output])

for i, layer in enumerate(int_model.layers):
    layer.name = 'layer_' + str(i)

train_x = np.load("train_x.npy")

predictions = model.predict(train_x)

# Randomly sample a number of positive and negative items to train and test the ruleset on.

positives = np.argwhere(predictions > 0.9)
positives = positives[:,0].reshape(positives.shape[0])
negatives = np.argwhere(predictions < 0.1)
negatives = negatives[:,0].reshape(negatives.shape[0])

positives = np.random.choice(positives,size=15,replace=True)
negatives = np.random.choice(negatives,size=15,replace=True)

total = np.concatenate((positives,negatives))

print(positives)
print(negatives)

# Also import the rest of the signals so that we can better visualise everything.

indices1 = [1880, 2839, 3225, 5082, 5401, 5763, 7802, 8139, 9210, 10047, 11343, 12701, 13039, 16147, 16523, 17354, 17754, 19080]
indices2 = [1895, 2854, 3240, 5097, 5416, 5777, 7817, 8151, 9223, 10060, 11357, 12716, 13052, 16162, 16537, 17369, 17765, 19095]

# This file contains the downsampled data.
data = pd.read_csv('reduced_data_train')

# The first column is an artifact; remove it.
data = data.iloc[:, 1:33]

# Extract all signals and signal names that matter. The setpoint and rotation
# speed values are kept apart.
signalnames = data.columns.to_numpy()
signalnames = signalnames[0:30]
signalnames = np.delete(signalnames, 10)
signals = data.to_numpy()
differentsignals = signals[:, np.array([10, 30])]
signals = signals[:, 0:30]
signals = np.delete(signals, 10, 1)

maxs = np.load('maxs.npy')
mins = np.load('mins.npy')
means = np.load('means.npy')
stds = np.load('stds.npy')


# Normalise the signals according to Z-score.
for i in range(29):
    if stds[i] != 0:
        signals[:, i] = (signals[:, i] - means[i]) / stds[i]
    else:
        signals[:, i] = np.zeros(len(total_predictions))

total_predictions = np.zeros(data.shape[0])

# Iterate over each timestep. If it is part of the test set it is given
# the corresponding value; otherwise, its prediction is set to 0 by default.
i = 0
j = 1880
k = 0
for i in range(len(predictions)):
    total_predictions[j] = predictions[i]
    j = j + 1
    if j == indices2[k] and k < 17:
        k = k + 1
        j = indices1[k]
        
# Preparatory steps finished; ruleset extraction itself starts here.

# Phase 1: collect all value segments and shapelets for the positive items
for index in total:

    if index in positives:
        type = 0
    else:
        type = 1

    item = train_x[index]
    for j in range(nsignals):
        if np.std(item[:, j]) < 0.01:
            item[:, j] = item[:, j] + (np.random.rand(ntimesteps) - 0.5) / 100

    # Load the statistics for all signals. Max and min values for each signal are
    # normalised using Z normalisation.
    maxs = np.load('maxs.npy')
    mins = np.load('mins.npy')
    means = np.load('means.npy')
    stds = np.load('stds.npy')
    for sign in range(nsignals):
        maxs[sign] = (maxs[sign] - means[sign]) / stds[sign]
        mins[sign] = (mins[sign] - means[sign]) / stds[sign]

    # Find the perturbed signal s.t. the prediction probability is changed to 0 (positive instance) or 1 (negative instance).
    prototype, tipping_point = perturbation(item,model,25,ntimesteps,nsignals,maxs,mins,type)

    # Find the most interesting signals. This is only required when there are too many signals to display in an explanation.
    differences = prototype - item
    differencesabs = np.abs(differences)
    totaldiff = np.zeros(nsignals)
    for i in range(nsignals):
        totaldiff[i] = np.sum(differencesabs[:, i])
    ranking = np.argsort(-totaldiff)
    
    # Select the four most interesting signals.
    selection = ranking[0:4]
    item2 = item[:, selection]
    prototype2 = prototype[:, selection]
    signals2 = signals[:, selection]
    signalnames2 = signalnames[selection]

    # Find the index in the total signals to which this index corresponds. This helps for visualisation.
    index2 = 1880
    help = 0
    for i in range(index):
        index2 = index2+1
        if index2 in indices2:
            help = help + 1
            index2 = indices1[help]

    if type == 0:
        visualise_pert("pos{}".format(index),prototype2,item2,index2,ntimesteps,len(total_predictions),signals2,differentsignals,4,signalnames2,total_predictions,3000)
    else:
        visualise_pert("neg{}".format(index), prototype2, item2, index2, ntimesteps, len(total_predictions), signals2,
                       differentsignals, 4,signalnames2, total_predictions, 3000)

    # Find all segments for which the change in values between orig and perturbed signal is significant.
    # We only look at the four most salient signals for oversight.
    if type == 0:
        segmentstemp, threshold = find_value_segments(prototype2,item2,ntimesteps,4,2,ranking)
    else:
        segmentstemp, threshold = find_value_segments(item2,prototype2,ntimesteps,4,2,ranking)

    for segm in segmentstemp:
        segm.item = index
        print("this temp segment starts at {}".format(segm.start))
    segments.extend(segmentstemp)
    
    # Next, we find all important shapelets. 

    # Find the four most relevant intermediate nodes for the original item.
    item_scores = int_model.predict(item.reshape(1, ntimesteps,nsignals))[0]
    item_ranking = np.argsort(-item_scores)
    relevant_nodes_item = item_ranking[:4]

    # Do the same for the perturbed item.
    prototype_scores = int_model.predict(prototype.reshape(1, ntimesteps,nsignals))[0]
    prototype_ranking = np.argsort(-prototype_scores)
    relevant_nodes_prototype = prototype_ranking[:4]

    # Remove all the nodes that are relevant for both items.
    for node in relevant_nodes_item:
        if node in relevant_nodes_prototype:
            relevant_nodes_item = np.delete(relevant_nodes_item, np.argwhere(relevant_nodes_item == node))
            relevant_nodes_prototype = np.delete(relevant_nodes_prototype,
                                                 np.argwhere(relevant_nodes_prototype == node))

    # Find all shapelets that are picked out by one or more intermediate nodes; distinguish between shapelets that are
    # present in the original item and those that are present in the perturbed item.
    shapelets_item = find_shapelets(item, relevant_nodes_item, int_model, ntimesteps, nsignals, 100)
    shapelets_prototype = find_shapelets(prototype, relevant_nodes_prototype, int_model, ntimesteps, nsignals, 100)

    # We need to swap the item and prototype shapelets if the item in question is not an anomaly.
    if type == 0:
        temp = shapelets_item
        shapelets_item = shapelets_prototype
        shapelets_prototype = temp

    # Correctly label all shapelets.
    for sli in shapelets_item:
        sli.item = index
        sli.type = 'orig'
    shapelets_orig.extend(shapelets_item)
    for slp in shapelets_prototype:
        slp.item = index
        slp.type = 'new'
    shapelets_new.extend(shapelets_prototype)

# Phase 2: further selecting the candidate conditions identified in phase 1. This phase consists of four steps.

# Step 1: separate value segments into higher than and lower than
i = 0
for segment in segments:
    print("this segment starts at {}".format(segment.start))
    if segment.type == 'higher':
        segmentshigher.append(segment)
    else:
        segmentslower.append(segment)
    i = i + 1

higherarrays = np.zeros((nsignals,ntimesteps))
lowerarrays = np.zeros((nsignals,ntimesteps))

# Step 2: calculate into how many higher / lower segments each timestep falls
for signal in range(nsignals):
    for timestep in range(ntimesteps):
        for segment in segmentshigher:
            if segment.signal == signal and segment.start <= timestep < segment.end:
                higherarrays[signal,timestep] = higherarrays[signal,timestep] + 1
        for segment in segmentslower:
            if segment.signal == signal and segment.start <= timestep < segment.end:
                lowerarrays[signal,timestep] = lowerarrays[signal,timestep] + 1

print("Higher arrays:")
print(higherarrays)

# Step 3: select potential V-conditions. These consist of timesteps that have sufficient
# support from individual value segments.
conditions = []
for signal in range(nsignals):
    switch = 0
    # add all possible higher-than segments with enough support
    intermediarysupport = support
    for i in range(ntimesteps):
        if switch == 0 and higherarrays[signal,i] >= intermediarysupport:
            cond = value_condition(type='higher',signal=signal,start=i)
            switch = 1
            highestsupport = higherarrays[signal,i]
        if switch == 1 and higherarrays[signal,i] > highestsupport:
            cond.start = i
            highestsupport = higherarrays[signal,i]
        if switch == 1 and (higherarrays[signal,i] < highestsupport):
            cond.end = i
            conditions.append(cond)
            intermediarysupport = higherarrays[signal,i] + 1
            switch = 0
        if switch == 1 and i == ntimesteps - 1:
            cond.end = i+1
            conditions.append(cond)
        if switch == 0 and higherarrays[signal,i] < support:
            intermediarysupport = support

    # do the same for lower-than segments
    switch = 0
    intermediarysupport = support
    for i in range(ntimesteps):
        if switch == 0 and lowerarrays[signal,i] >= intermediarysupport:
            cond = value_condition(type='lower',signal=signal,start=i)
            switch = 1
            highestsupport = lowerarrays[signal,i]
        if switch == 1 and lowerarrays[signal,i] > highestsupport:
            cond.start = i
            highestsupport = lowerarrays[signal,i]
        if switch == 1 and (lowerarrays[signal,i] < highestsupport):
            cond.end = i
            conditions.append(cond)
            intermediarysupport = lowerarrays[signal,i] + 1
            switch = 0
        if switch == 1 and i == ntimesteps-1:
            cond.end = i+1
            conditions.append(cond)
        if switch == 0 and lowerarrays[signal,i] < support:
            intermediarysupport = support

print("all value conditions:")
for condition in conditions:
    print(condition.type)
    print(condition.start)
    print(condition.end)

# Step 4: select potential shapelet conditions. First we have to define an auxiliary function that check if two shapelets
# are sufficiently similar via sliding window.

def compare_shapelets(sl1,sl2,threshold,check):
    if sl1.signal != sl2.signal:
        return (-1,-1)
        
    # Identify the longest shapelet

    if sl1.length >= sl2.length:
        longest = sl1
        shortest = sl2
    else:
        longest = sl2
        shortest = sl1
    
    if check == 1 and (longest.length-shortest.length) > longest.length / 2:
        return (-1,-1)
        

    # First, add average padding to the longest array to be sure.
    av = np.mean(longest.shape)
    array = np.array([av] * int(longest.length / 5))
    array = np.concatenate((array,longest.shape))
    array = np.concatenate((array, np.array([av] * int(longest.length / 5))))

    best_difference = 2
    best_index = -1
    
    # Next, use sliding window to see how well the shorter array matches the longer array.
    for i in range(0,len(array)-shortest.length+1):
        difference = np.sum(np.abs(array[i:i+shortest.length]-shortest.shape)) / shortest.length
        if difference < best_difference:
            best_difference = difference
            best_index = i

    if best_difference < threshold:
        return (1,best_index)
    else:
        return (-1,-1)

s_conditions_orig = []
s_conditions_new = []

# Next, we combine possible shapelets within the original shapelet set into 
# common conditions.

for sli in shapelets_orig:
    check = 0
    
    # We first go over all shapelet conditions already added to see if they correspond
    # to this shapelet.
    for scond in s_conditions_orig:
        success, ind = compare_shapelets(sli, scond, 0.25, 1)
        # Check if the new shapelet matches the existing condition.
        if success == 1:
            check = 1
            if sli.length > scond.length:
                longest = sli
                shortest = scond
            else:
                longest = scond
                shortest = sli

            # Option 1: If the begin index is within the zero padding, we start at the beginning of the long segment.
            if ind < int(longest.length / 5):
                start_index_long = 0
                start_index_short = int(longest.length / 5) - ind
            # Option 2: The begin index is not within the zero padding; we start at the index minus the length of the zero padding
            else:
                start_index_long = ind - int(longest.length / 5)
                start_index_short = 0

            # Option 1: The end index is within the zero padding; we stop at the end of the long segment.
            if ind + shortest.length > int(longest.length / 5) + longest.length:
                end_index_long = longest.length
                end_index_short = longest.length + int(longest.length / 5) - ind

            # Option 2: The end index is not within the zero padding; we stop at the index minus the length of the zero padding plus the short length
            else:
                end_index_long = ind + shortest.length - int(longest.length / 5)
                end_index_short = shortest.length

            # The new shape is the part of the longer segment in between the beginning and the end index, averaged with the new segment.
            if sli.length > scond.length:
                shape1 = sli.shape[start_index_long:end_index_long]
                shape2 = scond.shape[start_index_short:end_index_short]
            else:
                shape1 = sli.shape[start_index_short:end_index_short]
                shape2 = scond.shape[start_index_long:end_index_long]

            # Averaging depends on how much support the shapelet condition already had.
            old_support = len(scond.items)
            if len(shape1) == len(shape2):
                newshape = (shape1 + old_support * shape2) / (old_support + 1)

                # We update the shapelet condition.
                new_sl = shapelet_condition(signal=scond.signal, length=end_index_short - start_index_short, shape=newshape, type='orig')
                s_conditions_orig.remove(scond)
                s_conditions_orig.append(new_sl)
                break
            else:
                print('length error')
                check = 0

    # If none of the shapelet conditions found so far matches the new shapelet, we add the new shapelet here.
    if check == 0:
        s_conditions_orig.append(
            shapelet_condition(signal=sli.signal, length=sli.length, shape=sli.shape,type='orig'))

# Next, do the same for the non-original shapelet conditions.
for sli in shapelets_new:
    check = 0
    for scond in s_conditions_new:
        success, ind = compare_shapelets(sli, scond, 0.25,1)
        if success == 1:
            check = 1
            if sli.length > scond.length:
                longest = sli
                shortest = scond
            else:
                longest = scond
                shortest = sli

            # Option 1: If the index is within the zero padding, we start at the beginning of the long segment.
            if ind < int(longest.length / 5):
                start_index_long = 0
                start_index_short = int(longest.length / 5) - ind
            # Option 2: The index is not within the zero padding; we start at the index minus the length of the zero padding
            else:
                start_index_long = ind - int(longest.length / 5)
                start_index_short = 0

            # Option 1: The end index is within the zero padding; we stop at the end of the long segment.
            if ind + shortest.length > int(longest.length / 5) + longest.length:
                end_index_long = longest.length
                end_index_short = longest.length + int(longest.length / 5) - ind

            # Option 2: The end index is not within the zero padding; we stop at the index minus the length of the zero padding plus the short length
            else:
                end_index_long = ind + shortest.length - int(longest.length / 5)
                end_index_short = shortest.length

            print("check if {} and {} have the same length".format(end_index_long - start_index_long,
                                                                   end_index_short - start_index_short))
            # The new shape is the part of the longer segment in between the beginning and the end index, averaged with the new segment.
            if sli.length > scond.length:
                shape1 = sli.shape[start_index_long:end_index_long]
                shape2 = scond.shape[start_index_short:end_index_short]
            else:
                shape1 = sli.shape[start_index_short:end_index_short]
                shape2 = scond.shape[start_index_long:end_index_long]

            old_support = len(scond.items)

            if len(shape1) == len(shape2):
                newshape = (shape1 + old_support * shape2) / (old_support + 1)

                # We update the shapelet condition.
                new_sl = shapelet_condition(signal=scond.signal, length=end_index_short - start_index_short, shape=newshape,type='new')
                s_conditions_new.remove(scond)
                s_conditions_new.append(new_sl)
                break
            else:
                print('length error')
                check = 0

    if check == 0:
        s_conditions_new.append(
            shapelet_condition(signal=sli.signal, length=sli.length, shape=sli.shape, type='new'))

# Phase 3: Now that we have all possible conditions, we preprocess them to feed into the search algorithm.

# For each unary value condition, calculate the combined tipping point.
# At the same time, create a dictionary for all conditions, as well as a dictionary linking each item to a set of conditions
conddictionary = {}
itemdictionary = {}
for i in range(len(conditions)):
    # Add condition to dictionary.
    cond = conditions[i]
    conddictionary[i] = cond
    
    if cond.type == 'higher':
        # Calculate the tipping point, which is in between the lowest positive and the highest 
        # negative average.
        lowest_average = 9999
        for item in negatives:
            average = np.mean(train_x[item][cond.start:cond.end,cond.signal])
            if average < lowest_average:
                lowest_average = average
        highest_average = -9999
        for item in positives:
            average = np.mean(train_x[item][cond.start:cond.end,cond.signal])
            if average > highest_average:
                highest_average = average
        cond.tippingaverage = (lowest_average + highest_average) / 2
        
        # Add the condition to the item dictionary of all items that meet it.
        
        for item in total:
            if np.mean(train_x[item][cond.start:cond.end,cond.signal]) > cond.tippingaverage:
                cond.items.append(item)
                check = 0
                for k in itemdictionary.keys():
                    if k == item:
                        itemdictionary[k].append(i)
                        check =1
                if check == 0:
                    itemdictionary[item] = [i]
                    
        conditions[i] = cond
       
    # Repeat the above procedure for lower segments.
    if cond.type == 'lower':
        highest_average = -9999
        for item in negatives:
            average = np.mean(train_x[item][cond.start:cond.end,cond.signal])
            if average > highest_average:
                highest_average = average
        lowest_average = 9999
        for item in positives:
            average = np.mean(train_x[item][cond.start:cond.end,cond.signal])
            if average < lowest_average:
                lowest_average = average
        cond.tippingaverage = (lowest_average + highest_average) / 2
        
        for item in total:
            if np.mean(train_x[item][cond.start:cond.end,cond.signal]) < cond.tippingaverage:
                cond.items.append(item)
                check = 0
                for k in itemdictionary.keys():
                    if k == item:
                        itemdictionary[k].append(i)
                        check =1
                if check == 0:
                    itemdictionary[item] = [i]
        
        conditions[i] = cond

intlength = len(conddictionary)

# Also add the shapelet conditions to the dictionaries
for i in range(len(s_conditions_orig)):
    cond = s_conditions_orig[i]
    conddictionary[i+intlength] = cond
    for shapelet in shapelets_orig:
        success, other = compare_shapelets(cond,shapelet,0.25,1)
        if shapelet.item in negatives and success == 1:
            cond.items.append(shapelet.item)
            if shapelet.item in itemdictionary.keys():
                if not(i + intlength in itemdictionary[shapelet.item]):
                    itemdictionary[shapelet.item].append(i + intlength)
            else:
                itemdictionary[shapelet.item] = [i+intlength]
    for shapelet in shapelets_new:
        success,other = compare_shapelets(cond,shapelet,0.25,1)
        if shapelet.item in positives and success == 1:
            cond.items.append(shapelet.item)
            if shapelet.item in itemdictionary.keys():
                if not(i + intlength in itemdictionary[shapelet.item]):
                    itemdictionary[shapelet.item].append(i + intlength)
            else:
                itemdictionary[shapelet.item] = [i+intlength]

intlength = len(conddictionary)

for i in range(len(s_conditions_new)):
    cond = s_conditions_new[i]
    conddictionary[i+intlength] = cond
    for shapelet in shapelets_new:
        success, other = compare_shapelets(cond,shapelet,0.25,1)
        if shapelet.item in negatives and success == 1:
            cond.items.append(shapelet.item)
            if shapelet.item in itemdictionary.keys():
                if not(i + intlength in itemdictionary[shapelet.item]):
                    itemdictionary[shapelet.item].append(i + intlength)
            else:
                itemdictionary[shapelet.item] = [i+intlength]
    for shapelet in shapelets_orig:
        success,other = compare_shapelets(cond,shapelet,0.25,1)
        if shapelet.item in positives and success == 1:
            cond.items.append(shapelet.item)
            if shapelet.item in itemdictionary.keys():
                if not(i + intlength in itemdictionary[shapelet.item]):
                    itemdictionary[shapelet.item].append(i + intlength)
            else:
                itemdictionary[shapelet.item] = [i+intlength]

#plot_conditions(conditions,ntimesteps,nsignals)
print(conddictionary)
print(itemdictionary)

# Phase 4. We now have a dictionary of all conditions, and a dictionary of all items
# telling us to which conditions they correspond. The next step is to preselect those
# combinations of conditions that correspond with most items. This gives us a set of
# possible rules.

# Use apriori to find possible combined conditions
possible_conditions = list(apriori(itemdictionary.values(),min_support=0.35))
rules = [list(element.items) for element in possible_conditions]
possible_rules = [[conddictionary[i] for i in rule] for rule in rules]

# Phase 5. Use a search algorithm to find the subset of possible rules that
# best performs on the various performance criteria.

# Auxiliary function to determine if an item matches a specific rule:
def check_item(rule,item):
    for condition in rule:
        if isinstance(condition,value_condition):
            #item2 = train_x[item]
            #if condition.type == 'higher':
                #if np.average(item2[condition.start:condition.end,condition.signal]) <= condition.tippingaverage:
                    #return 0
            #if condition.type =='lower':
                #if np.average(item2[condition.start:condition.end,condition.signal]) >= condition.tippingaverage:
                    #return 0
            if not(item in condition.items):
                return 0
        else:
            if not (item in condition.items):
                return 0
    return 1

# This function evaluates a ruleset on the objective function.
def evaluate_ruleset(ruleset,maxdisagreement,maxoverlap,maxsize,maxmaxwidth,normal_items,anom_items,l1,l2,l3,l4,l5,iteration):
    # Calculate the disagreement.
    disagreement = 0
    for i in normal_items:
        for rule in ruleset:
            if check_item(rule,i):
                disagreement = disagreement+1

    # Calculate overlap
    overlap = 0
    for i in range(len(ruleset)):
        for j in range(i,len(ruleset)):
            for item in anom_items:
                if check_item(ruleset[i],item) and check_item(ruleset[j],item):
                    overlap = overlap + 1

    # Calculate cover
    cover = 0
    for i in anom_items:
        check = 0
        for rule in ruleset:
            if check_item(rule,i):
                check = 1
        cover = cover + check

    # Calculate size
    size = len(ruleset)

    # Calculate maxwidth
    maxwidth = 0
    for rule in ruleset:
        if len(rule) > maxwidth:
            maxwidth = len(rule)

    # Subtract disagreement from max disagreement (i.e. each rule disagrees with each normal item)
    f1 = maxdisagreement - disagreement

    # Subtract overlap from max overlap (i.e. any two rules overlap)
    f2 = maxoverlap - overlap

    # Subtract size from maxsize
    f3 = maxsize - size

    # Subtract maxwidth from maxmaxwidth
    f4 = maxmaxwidth - maxwidth
    
    if iteration % 1000 == 0:
        print("l1: {}, f1: {}".format(l1,f1))
        print("l2: {}, f2: {}".format(l2,f2))
        print("l3: {}, f3: {}".format(l3,f3))
        print("l4: {}, f4: {}".format(l4,f4))
        print("l5: {}, f5: {}".format(l5,cover))

    # return the total weighted function
    return l1 * f1 + l2 * f2 + l3 * f3 + l4* f4 + l5 * cover

# A neighbour function which randomly perturbs a ruleset by
# a) Adding a rule
# b) Removing a rule (if possible)
# c) Swapping a rule for a different rule
def neighbour(ruleset,possible_rules):
    rand = np.random.randint(3)
    neighbour = ruleset.copy()
    if rand == 0 and len(neighbour) < len(possible_rules):
        newrule = possible_rules[np.random.randint(len(possible_rules))]
        while newrule in neighbour:
            newrule = possible_rules[np.random.randint(len(possible_rules))]
        neighbour.append(newrule)
    if rand == 1 and len(neighbour) > 1:
        del neighbour[np.random.randint(len(neighbour))]
    if rand == 2:
        index = np.random.randint(len(neighbour))
        newrule = possible_rules[np.random.randint(len(possible_rules))]
        while newrule in neighbour and newrule != neighbour[index]:
            newrule = possible_rules[np.random.randint(len(possible_rules))]
        neighbour[index] = newrule
    return neighbour

def local_search(ruleset,possible_rules,iterations):
    
    # Calculate the maximum disagreement
    maxdisagreement = len(possible_rules) * len(positives)
    
    # Calculate the maximum overlap
    maxoverlap = len(negatives) * len(possible_rules) * len(possible_rules)

    # Calculate the maximum maximum width
    maxmaxwidth = 0
    for rule in possible_rules:
        if len(rule) > maxmaxwidth:
            maxmaxwidth = len(rule)

    # We set all the parameters inversely proportional to the max value of the corresponding term.
    l1 = 1 / (len(possible_rules) * len(positives))
    l2 = 1 / (len(possible_rules) * len(possible_rules) * len(negatives))
    l3 = 1 / len(possible_rules) 
    l4 = 1 / maxmaxwidth
    l5 = 1 / len(negatives)
    
    temperature = 10

    for i in range(iterations):

        nb = neighbour(ruleset.copy(),possible_rules)
        ruleset_score = evaluate_ruleset(ruleset,maxdisagreement,maxoverlap,len(possible_rules),maxmaxwidth,positives,negatives,l1,l2,l3,l4,l5,i)
        neighbour_score = evaluate_ruleset(nb,maxdisagreement,maxoverlap,len(possible_rules),maxmaxwidth,positives,negatives,l1,l2,l3,l4,l5,i)
        if i % 1000 == 0:
            print(i)
            print(ruleset_score)
            print(neighbour_score)
            

        # We use simulated annealing to arrive at the best candidate ruleset.
        if neighbour_score >= ruleset_score:
            ruleset = nb
        else:
            diff = neighbour_score - ruleset_score
            prob = math.exp(diff/temperature)
            if random.random() < prob:
                ruleset = nb
        
        temperature = 0.999 * temperature

    return ruleset

print("Possible ruleset:")
print(possible_rules)
initial_ruleset = [possible_rules[np.random.randint(len(possible_rules))]]
final_ruleset = local_search(initial_ruleset,possible_rules,10000)

print("A sequence likely precedes a failed ignition attempt if one of the following rules hold:")
for i in range(len(final_ruleset)):
    print("{}.".format(i+1),end=" ")
    rule = final_ruleset[i]
    for j in range(len(rule)):
        condition = rule[j]
        if isinstance(condition,value_condition):
            print("signal {} has a value {} than {} between {} and {}".format(condition.signal,condition.type,condition.tippingaverage,condition.start,condition.end),end="")
        else:
            if condition.type == 'orig':
                print("signal {} contains the following shapelet:".format(condition.signal))
                plt.clf()
                fig, (ax1) = plt.subplots(1, 1, figsize=(10, 3))
                ax1.plot(list(range(condition.length)), condition.shape)
                plt.savefig("shapelet_rule{}".format(i))
            if condition.type == 'new':
                print("signal {} does not contain the following shapelet:".format(condition.signal))
                plt.clf()
                fig, (ax1) = plt.subplots(1, 1, figsize=(10, 3))
                ax1.plot(list(range(condition.length)), condition.shape)
                plt.savefig("shapelet_rule{}".format(i))
        if j + 1 < len(rule):
            print(" and",end=" ")
    print(" ")
    
#print(positives)
#print(negatives)