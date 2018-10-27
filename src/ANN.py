import numpy as np
import time
import datetime
from sklearn.feature_extraction.text import CountVectorizer

labelDict = {'transport': 0, 'walking': 1, 'airplane': 2}
classes = labelDict.keys()

# compute sigmoid nonlinearity
def sigmoid(x):
    output = 1 / (1 + np.exp(-x))
    return output


# convert output of sigmoid function to its derivative
def sigmoid_output_to_derivative(output):
    return output * (1 - output)


def train(X, y, hidden_neurons=10, alpha=1, epochs=50000, dropout=False, dropout_percent=0.5):
    print("Training with %s neurons, alpha:%s, dropout:%s %s" % (
    hidden_neurons, str(alpha), dropout, dropout_percent if dropout else ''))
    print("Input matrix: %sx%s    Output matrix: %sx%s" % (len(X), len(X[0]), 1, len(classes)))
    np.random.seed(1)

    last_mean_error = 1
    # randomly initialize our weights with mean 0
    synapse_0 = 2 * np.random.random((len(X[0]), hidden_neurons)) - 1
    synapse_1 = 2 * np.random.random((hidden_neurons, len(classes))) - 1

    prev_synapse_0_weight_update = np.zeros_like(synapse_0)
    prev_synapse_1_weight_update = np.zeros_like(synapse_1)

    synapse_0_direction_count = np.zeros_like(synapse_0)
    synapse_1_direction_count = np.zeros_like(synapse_1)

    for j in iter(range(epochs + 1)):

        # Feed forward through layers 0, 1, and 2
        layer_0 = X
        layer_1 = sigmoid(np.dot(layer_0, synapse_0))

        if (dropout):
            layer_1 *= np.random.binomial([np.ones((len(X), hidden_neurons))], 1 - dropout_percent)[0] * (
                    1.0 / (1 - dropout_percent))

        layer_2 = sigmoid(np.dot(layer_1, synapse_1))

        # how much did we miss the target value?
        layer_2_error = y - layer_2

        if (j % 1000) == 0 and j > 500:
            # if this 10k iteration's error is greater than the last iteration, break out
            # if np.mean(np.abs(layer_2_error)) < last_mean_error:
            print("delta after " + str(j) + " iterations:" + str(np.mean(np.abs(layer_2_error))))
            last_mean_error = np.mean(np.abs(layer_2_error))
            # else:
            #     print("break:", np.mean(np.abs(layer_2_error)), ">", last_mean_error)
            #     break

        # in what direction is the target value?
        # were we really sure? if so, don't change too much.
        layer_2_delta = layer_2_error * sigmoid_output_to_derivative(layer_2)

        # how much did each l1 value contribute to the l2 error (according to the weights)?
        layer_1_error = layer_2_delta.dot(synapse_1.T)

        # in what direction is the target l1?
        # were we really sure? if so, don't change too much.
        layer_1_delta = layer_1_error * sigmoid_output_to_derivative(layer_1)

        synapse_1_weight_update = (layer_1.T.dot(layer_2_delta))
        synapse_0_weight_update = (layer_0.T.dot(layer_1_delta))

        if (j > 0):
            synapse_0_direction_count += np.abs(
                ((synapse_0_weight_update > 0) + 0) - ((prev_synapse_0_weight_update > 0) + 0))
            synapse_1_direction_count += np.abs(
                ((synapse_1_weight_update > 0) + 0) - ((prev_synapse_1_weight_update > 0) + 0))

        synapse_1 += alpha * synapse_1_weight_update
        synapse_0 += alpha * synapse_0_weight_update

        prev_synapse_0_weight_update = synapse_0_weight_update
        prev_synapse_1_weight_update = synapse_1_weight_update

    # now = datetime.datetime.now()

    # # persist synapses
    # synapse = {'synapse0': synapse_0.tolist(), 'synapse1': synapse_1.tolist(),
    #            'datetime': now.strftime("%Y-%m-%d %H:%M"),
    #            'words': words,
    #            'classes': classes
    #            }
    # synapse_file = "synapses.json"
    #
    # with open(synapse_file, 'w') as outfile:
    #     json.dump(synapse, outfile, indent=4, sort_keys=True)
    # print("saved synapses to:", synapse_file)


dataFileName = "data_updated.txt"

with open(dataFileName) as f:
    content = f.readlines()

content = [x.strip() for x in content]

labels = [x.split(",")[-1] for x in content]
data = [','.join(x.split(",")[0:-1]) for x in content]


def tokenize(txt):
    return txt.split(",")


vec = CountVectorizer(tokenizer=tokenize)
training = vec.fit_transform(data).toarray()

output = []
for lbl in labels:
    output_row = list([0]*3)
    output_row[labelDict.get(lbl)] = 1
    output.append(output_row)

X = np.array(training[0:5000])
y = np.array(output[0:5000])

start_time = time.time()

train(X, y, hidden_neurons=20, alpha=0.1, epochs=10000, dropout=False, dropout_percent=0.2)

elapsed_time = time.time() - start_time
print("processing time:", elapsed_time, "seconds")