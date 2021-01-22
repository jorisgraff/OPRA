import keras
from keras.models import model_from_json
from functions import visualise_LIME

import pandas as pd
import numpy as np
import lime.lime_tabular

test_x = np.load('test_x.npy')
ntimesteps = test_x.shape[1]
nsignals = test_x.shape[2]

X = test_x.shape[0]
Y = test_x.shape[1] * test_x.shape[2]

test_x_flattened = np.empty((X,Y))

for i in range(X):
    test_x_flattened[i] = test_x[i].reshape((1,Y))

# load json and create model
json_file = open('modelsmall0.35.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
# load weights into new model
model.load_weights("smallmodel0.35.h5")

def predict_flat_instances(flat_array):
    # load json and create model
    json_file = open('modelsmall0.35.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    # load weights into new model
    model.load_weights("smallmodel0.35.h5")

    X = flat_array.shape[0]

    array = flat_array.reshape((X,ntimesteps,nsignals))

    prediction = model.predict(array)
    negprediction = 1 - prediction

    totalpredictions = np.empty((prediction.shape[0],2))

    for i in range(prediction.shape[0]):
        totalpredictions[i,0] = prediction[i]

        totalpredictions[i,1] = negprediction[i]

    print(totalpredictions)

    return(totalpredictions)

indices1 = [427,5285,19182,19531,20318,20701]
indices2 = [439,5298,19195,19546,20331,20714]

for index in [430,437,5289,19183,19192,19535,19542,20322,20330,20711]:

    index1 = index
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

    test_item = test_x_flattened[index2]
    test_item_orig = test_x[index2]

    explainer = lime.lime_tabular.LimeTabularExplainer(test_x_flattened,'classification',discretize_continuous=False,verbose=True)
    exp = explainer.explain_instance(test_item,predict_flat_instances,num_features=ntimesteps*nsignals)

    exp = exp.as_list()

    exp_array = np.empty(nsignals*ntimesteps)

    for item in exp:
        exp_array[int(item[0])] = item[1]

    exp_array = exp_array.reshape((ntimesteps,nsignals))

    threshold = np.std(exp_array)

    relevance = np.abs(exp_array)

    count = 0

    for i in range(ntimesteps):
        for j in range(nsignals):
            if relevance[i,j]>=threshold:
                count = count+1
    proportion = count/(ntimesteps*nsignals)

    output_file = open("LIME_output {}.txt".format(index1),"w")
    output_file.write("The number of relevant steps is {} which is {} of total".format(count,proportion))
    output_file.close()
            
#totalrelevance = np.zeros(29)
#for i in range(29):
    #totalrelevance[i] = np.sum(relevance[:,i])
#ranking = np.argsort(totalrelevance)
#selection = ranking[0:4]

# This file contains the downsampled data.
#data = pd.read_csv('reduced_data')

# The first column is an artifact; remove it.
#data = data.iloc[:, 1:33]

# Extract all signals and signal names that matter. The setpoint and rotation
# speed values are kept apart.
#signalnames = data.columns.to_numpy()
#signalnames = signalnames[0:30]
#signalnames = np.delete(signalnames, 10)
#signals = data.to_numpy()
#differentsignals = signals[:, np.array([10, 30])]
#signals = signals[:, 0:30]
#signals = np.delete(signals, 10, 1)

#stds = np.load('stds.npy')
#means = np.load('means.npy')

# Normalise the signals according to Z-score.
#for i in range(29):
    #if stds[i] != 0:
        #signals[:,i] = (signals[:,i] - means[i]) / stds[i]
    #else:
        #signals[:,i] = np.zeros(len(total_predictions))


#predictions = model.predict(test_x)

#total_predictions = np.zeros(data.shape[0])

# Iterate over each timestep. If it is part of the test set it is given
# the corresponding value; otherwise, its prediction is set to 0 by default.
#i = 0
#j = 427
#k = 0
#for i in range(len(predictions)):
    #total_predictions[j] = predictions[i]
    #j = j + 1
    #if j == indices2[k] and k < 5:
        #k = k + 1
        #j = indices1[k]

#item2 = test_item_orig[:, selection]
#signals2 = signals[:, selection]
#signalnames2 = signalnames[selection]
#exp_array2 = exp_array[:,selection]

#visualise_LIME(item2,exp_array2,4,signals2,signalnames2,total_predictions,index1,test_x.shape[1],len(total_predictions))
