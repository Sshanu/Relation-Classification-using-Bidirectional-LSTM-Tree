## Relation Classification using LSTMs on Sequences and Tree Structures

We implemented a architecture based on the paper [End-to-End Relation Extraction using LSTMs
on Sequences and Tree Structures](http://www.aclweb.org/anthology/P/P16/P16-1105.pdf). This recurrent neural network based model captures both word sequence and dependency tree substructure information by stacking bidirectional treestructured LSTM-RNNs on bidirectional sequential LSTM-RNNs. This allows our model to jointly represent both entities and relations with shared parameters in a single model.


Our model allows
joint modeling of entities and relations in a single
model by using both bidirectional sequential
(left-to-right and right-to-left) and bidirectional
tree-structured (bottom-up and top-down) LSTMRNNs.


## Model 
The model mainly consists of three representation layers:
a embeddings layer, a word sequence based LSTM-RNN layer (sequence layer), and finally a dependency subtree based LSTM-RNN layer (dependency layer).

### Embedding Layer
Embedding layer consists of words, part-of-speech (POS) tags, dependency relations.

### Sequence Layer
The sequence layer represents words in a linear sequence
using the representations from the embedding layer. We represent the word sequence in a sentence with bidirectional LSTM-RNNs. 
The LSTM unit at t-th word receives the concatenation of word and POS embeddings as its input vector. 

<p align="center">
  <img src="/img/lstm_seq.jpg">
</p>

We also concatenate the hidden state vectors of the two directions’ LSTM units corresponding to each word (denoted as −→ht and ←−ht) as its output vector (st), and pass it to the subsequent layers.

### Entity Detection 
We perform entity detection on top of the sequence
layer. We employ a two-layered NN with an hidden layer and a softmax output layer for entity detection.

### Dependency Layer

![Relation Classification Network](/img/lstm_tree.jpg)

Model | Train-Accuracy | Test-Accuracy| Epochs
--- | --- | ---| ---
model3v1 | 97.54 | 66.5 | 11
model3v2 | 99.9 | 70.69 | 19


* Learning rate = 0.001 
* Learning rate decay = 0.96
* state size = 100
* lambda_l2 = 0.0001
* Gradient Clipping = 10
* Entity Detection Pretrained


### [model3v1](https://github.com/Sshanu/Relation-Classification/blob/master/LSTM%20Seq%20and%20Tree/model3v1.ipynb)
* Bidirectional LSTM over whole sentence
* Bottom-up and Top-down LSTM along Shortest Dependency Path with childrens from Dependency tree.

### [model3v2](https://github.com/Sshanu/Relation-Classification/blob/master/LSTM%20Seq%20and%20Tree/model3v2.ipynb)
* Dropout on hidden layers of both entity detection and relation classifier of 0.3.
