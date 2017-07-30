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

![Relation Classification Network](/img/lstm_tree.jpg)

### Embedding Layer
Embedding layer consists of words, part-of-speech (POS) tags, dependency relations.

### Sequence Layer
The sequence layer represents words in a linear sequence
using the representations from the embedding layer. We represent the word sequence in a sentence with bidirectional LSTM-RNNs. 
The LSTM unit at t-th word receives the concatenation of word and POS embeddings as its input vector. 

<p align="center">
  <img src="/img/lstm_seq.jpg">
</p>

We also concatenate the hidden state vectors of the two directions’ LSTM units corresponding to each word (denoted as ↑ht and ↓ht) as its output vector (st), and pass it to the subsequent layers.

### Entity Detection 
We perform entity detection on top of the sequence
layer. We employ a two-layered NN with an hidden layer and a softmax output layer for entity detection.

### Dependency Layer
The dependency layer represents a relation between a pair of two target words (corresponding to a relation candidate in relation classification) in
the dependency tree.

This layer mainly focuses on the shortest path between a pair of target words in the dependency tree (i.e., the path between the least common node and the two target words).

We employ bidirectional tree-structured LSTMRNNs (i.e., bottom-up and top-down) to represent a relation candidate by capturing the dependency
structure around the target word pair. This bidirectional structure propagates to each node not only the information from the leaves but also information from the root. This is especially important for relation classification, which makes use of argument nodes near the bottom of the tree, and our top-down LSTM-RNN sends information from the top of the tree to such near-leaf nodes (unlike in standard bottom-up LSTM-RNNs).

Tree-structured LSTM-RNN's equations :
<p align="center">
  <img src="/img/lstm_tree_eq.jpg">
</p>

While we use one node from Shortest Dependency path, then the hidden and current states of the children of this node in Dependency Tree are taken as previous state in LSTM.

We stack the dependency layers (corresponding to relation candidates) on top of the sequence layer to incorporate both word sequence and dependency tree structure information into the output.
The dependency-layer LSMT unit at the t-th word recives as input, the concatenation of its corresponding hidden state vectors st in the sequence layer, dependency type embedding.

### Relation Classification 
The relation candidate vector is constructed as
the concatenation dp = [↑hpA; ↓hp1; ↓hp2], where ↑hpA is the hidden state vector of the top LSTM unit in the bottom-up LSTM-RNN (representing the lowest common ancestor of the target word pair p), and ↓hp1, ↓hp2 are the hidden state vectors of the two LSTM units representing the first and second target words in the top-down LSTMRNN.

Similarly to the entity detection, we employ a two-layered NN with an hidden layer and a softmax output layer.

### Training

We update the model parameters including weights, biases, and embeddings by BPTT and Adam gradient descent with gradient clipping, L2-regularization
(we regularize weights W and U, not the bias terms b). We also apply dropout to the embedding layer and to the final hidden layers for entity detection and relation classification. We employ entity pretraining to improve the model.

### Data

SemEval-2010 Task 8 defines 9 relation types between nominals and a tenth type Other when two nouns have none of these relations and no direction is considered.
## Experiments

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
