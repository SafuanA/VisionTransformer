from matplotlib import pyplot as plt
import numpy as np
import os

directory='plots'
if not os.path.exists(directory):
    os.makedirs(directory)
plot_counter=0

def plot_features_sequence(data, original_data, title='fbfeatures'):
    _, w = data.shape
    _, w_o = original_data.shape
    predicted_data = original_data.clone()
      
    predicted_data[:,w_o-w:] = data
    plot_features(predicted_data)
    plot_features(original_data, name="original_")

def plot_features(data, title='fbfeatures', name="predicted_", transpose=False):
    global plot_counter
    data = data.squeeze()
    data = data.cpu().detach()

    x_label = 'frame' 
    y_label = 'freq_bin'
    if transpose:
        x_label = 'freq_bin'
        y_label = 'frame'
        data = data.T
    plt.imshow(data, origin='upper')
    plt.gcf().subplots_adjust(bottom=0.15)

    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.savefig(directory +'/'+name + str(plot_counter)+'.png')
    plot_counter+=1

def plot_weights(weights):
    x = 1

def plot_score():
    labels, scores = np.loadtxt("score.txt").T

    target_score = []
    nontarget_score = []

    for idx,i in enumerate(labels):
        if i == 0:
            nontarget_score.append(scores[idx])
        else:
            target_score.append(scores[idx])

    print(scores.shape)
    print(labels.shape)

    plt.hist(target_score, bins=100, label="target score")
    plt.hist(nontarget_score, bins=100, label="nontarget score")
    plt.legend()
    plt.tight_layout()
    plt.savefig("test.png")

def get_class_distribution(dataset_obj):
    count_dict = {k:0 for k,v in dataset_obj.class_to_idx.items()}
    
    for element in dataset_obj:
        y_lbl = element[1]
        #y_lbl = idx2class[y_lbl]
        count_dict[y_lbl] += 1
            
    return count_dict