import matplotlib.pyplot as plt
import numpy as np

N = 2 # change to 6 and add LSTMU and LSTMB Models
###########################################################################
# Binary Graph Models
f1_means = (0.853993887, 0.811600324)
f1_err = (0.0161547238553041, 0.00414287866088785)
precision_means = (0.888551695, 0.68561981031667)
precision_err = (0.0418452561710614, 0.00116344084501155)
recall_means = (0.824293785310734, 0.994350282485875)
recall_err = (0.0337357474989223, 0.0106532095094021)
accuracy_means = (0.806589147286821, 0.683333333333333)
accuracy_err = (0.0232880912958136, 0.00518408337150901)

fig, ax = plt.subplots()

ind = np.arange(N) 
width = 0.10

ax.bar(ind, f1_means, width, label='F1', yerr=f1_err)
ax.bar(ind + width, precision_means, width,
    label='Precision', yerr=precision_err)
ax.bar(ind + 2*width, recall_means, width,
    label='Recall', yerr=recall_err)
ax.bar(ind + 3*width, accuracy_means, width,
    label='Accuracy', yerr=accuracy_err)

x_labels = ['CNN Pre-trained', 'CNN Post-trained']
ax.set_ylabel('Performance')
ax.set_xticks(ind + width + width/2)
ax.set_xticklabels(x_labels)
ax.set_xlabel('Models')
ax.set_title('Binary Model Performances')
ax.legend()
plt.grid(True, 'major', 'y', ls='--', lw=.5, c='k', alpha=.3)

plt.show()
##################################################################################
# Categorical Bar Graph
f1_means = (0.680436811, 0.397195236)
precision_means = (0.711238296411343, 0.40775712823729)
recall_means = (0.693798449612402, 0.512790697674418)
accuracy_means = (0.693798449612403, 0.512790697674418)
f1_err = (0.0399069890331443, 0.072036577433337)
precision_err = (0.0380529926709191, 0.0751936234383296)
recall_err = (0.0305194103643869, 0.0654734166653383)
accuracy_err = (0.0305194103643868, 0.0654734166653383)

fig, ax = plt.subplots()

ind = np.arange(N) 
width = 0.10

ax.bar(ind, f1_means, width, label='F1', yerr=f1_err)
ax.bar(ind + width, precision_means, width,
    label='Precision', yerr=precision_err)
ax.bar(ind + 2*width, recall_means, width,
    label='Recall', yerr=recall_err)
ax.bar(ind + 3*width, accuracy_means, width,
    label='Accuracy', yerr=accuracy_err)

x_labels = ['CNN Pre-trained', 'CNN Post-trained']
ax.set_ylabel('Performance')
ax.set_xticks(ind + width + width/2)
ax.set_xticklabels(x_labels)
ax.set_xlabel('Models')
ax.set_title('Categorical Model Performances')
ax.legend()
plt.grid(True, 'major', 'y', ls='--', lw=.5, c='k', alpha=.3)

plt.show()
########################################################################
# Binary Model Accuracies
epoch = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
acc_mean_cnn_gnv = [0.64055, 0.77237, 0.79342, 0.79146, 0.825275, 0.7989833333, 0.8301333333]
acc_mean_cnn_w2v = [0.58796, 0.59468, 0.5722777778, 0.5698142857, 0.5943, 0.5962]
plt.plot(epoch[0:len(acc_mean_cnn_gnv)], acc_mean_cnn_gnv, color='g')
plt.plot(epoch[0:len(acc_mean_cnn_w2v)], acc_mean_cnn_w2v, color='orange')
plt.xlabel('Models')
plt.ylabel('Validation Accuracy')
plt.title('Validation Accurcay of Binary Models')
plt.show()

##########################################################################
# Binary Model Losses
epoch = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
acc_mean_cnn_gnv = [0.6087, 0.38943, 0.36986, 0.37214, 0.368725, 0.3908833333, 0.4152]
acc_mean_cnn_w2v = [0.68566, 0.67853, 0.6842333333, 0.6857571429, 0.6787, 0.6758]
plt.plot(epoch[0:len(acc_mean_cnn_gnv)], acc_mean_cnn_gnv, color='g')
plt.plot(epoch[0:len(acc_mean_cnn_w2v)], acc_mean_cnn_w2v, color='orange')
plt.xlabel('Models')
plt.ylabel('Validation Losses')
plt.title('Validation Losses of Binary Models')
plt.show()
########################################################################
# Categorical Model Accuracies
epoch = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
acc_mean_cnn_gnv = [0.54895, 0.67026, 0.70169, 0.71043, 0.7130375, 0.7592571429, 0.748225, 0.7157666667]
acc_mean_cnn_w2v = [0.41598, 0.43175, 0.43496, 0.4342666667, 0.4330714286, 0.4952, 0.5524, 0.381]
plt.plot(epoch[0:len(acc_mean_cnn_gnv)], acc_mean_cnn_gnv, color='g')
plt.plot(epoch[0:len(acc_mean_cnn_w2v)], acc_mean_cnn_w2v, color='orange')
plt.xlabel('Models')
plt.ylabel('Validation Accuracy')
plt.title('Validation Accurcay of Categorical Models')
plt.show()

##########################################################################
# Categorical Model Losses
epoch = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
acc_mean_cnn_gnv = [0.92399, 0.66784, 0.65001, 0.65123, 0.5930375, 0.5650142857, 0.621275, 0.606]
acc_mean_cnn_w2v = [1.02797, 1.03056, 1.03521, 1.059077778, 1.053228571, 1.0162, 0.9681, 1.148]
plt.plot(epoch[0:len(acc_mean_cnn_gnv)], acc_mean_cnn_gnv, color='g')
plt.plot(epoch[0:len(acc_mean_cnn_w2v)], acc_mean_cnn_w2v, color='orange')
plt.xlabel('Models')
plt.ylabel('Validation Losses')
plt.title('Validation Losses of Categorical Models')
plt.show()




