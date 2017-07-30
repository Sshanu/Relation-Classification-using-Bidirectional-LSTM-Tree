## Relation Classification using LSTM Networks along Shortest Dependency Paths

First we implemented a architecture following a paper [Classifying Relations via Long Short Term Memory Networks along Shortest Dependency Paths](https://pdfs.semanticscholar.org/0f44/366c1e1446cfd51258c68bd1da14fe9c7f10.pdf?_ga=2.136229944.807016038.1498203433-264083776.1497442258) by Yan Xu and others. 
This neural architecture utilizes the shortest dependency path between two entities in a sentence. 
The shortest dependency paths retain most relevant information (to relation classification), while eliminating irrelevant words in the sentence.

## SDP-LSTM Model

![LCA Shortest Path](/img/lca.jpg)

First sentence is parsed to a dependency tree by the [Stanford parser](https://nlp.stanford.edu/software/stanford-dependencies.shtml), the shortest dependency path(SDP) is extracted as the input of our network.

Dependency trees are a kind of directed graph, so direction of relation matters. Hence we separate SDP into two sub-paths, each from an entity to the common ancestor node. Along the SDP, three different types of information(as channels) are used, including the words, POS tags, dependency types.
In each channel, e.g. words, are mapped to real-valued vectors, called embeddings, which capture the underlying meanings of the inputs.

### Channels

* Each word in a given sentence is mapped to a real-valued vector by looking up in a word embedding table of Glove (pretrained).
* Since word embeddings are obtained on a generic corpus of a large scale, the information they contain may not agree with a specific sentence. We deal with this problem by allying each input word with its POS tag, e.g., noun, verb, etc.
* The dependency types between words provide grammatical relationships in a sentence that can easily be understood and effectively used by people
without linguistic expertise
 Two recurrent neural networks pick up information along the left and right sub-paths of the SDP. 

### Recurrent Neural Networks

Recurrent Neural Networks have one problem, known as gradient vanishing or exploding problem. Long short term memory(LSTM) overcome this problem by introducing an adaptive gating mechanism, which keep the previous state and memorize the extracted features of the current data input.
LSTM-based recurrent neuralnetwork comprises four components: an input gate, a forget gate, an output gate, and a memory cell.
The two SDP-LSTM  propagate bottom-up from the entities to their common ancestor. This way, the model is direction-sensitive.

A max pooling layer packs, for each sub-path, the recurrent network’s states, to a fixed vector by taking the maximum value in each dimension.
The pooling layers from different channels are concatenated, and then connected to a hidden layer. Finally, we have a softmax output layer for
classification. 

The training objective is the cross-entropy error with l2-regulaizer to avoid overfitting. We apply Adam Gradient Descent for optmization.

## Experiments

Model | Train-Accuracy | Test-Accuracy| Epochs
--- | --- | ---| ---
modelv1 | 99.45 | 61.4 | 10
modelv2 | 100 | ? | 10
modelv3 | 84.03 | 60.4 | 20
modelv4 | 96.1 | 63.2 | 60
modelv5 | 92.2 | 62.3 | 60
modelv6 | 97.3 | 61.4 | 34
modelv7 | 94.6 | 60.03 | 20
modelv8 | 98.96 | 62.5 | 60

### [modelv1](https://github.com/Sshanu/Relation-Classification/blob/master/LCA%20Shortest%20Path/modelv1.ipynb)
* Learning rate = 0.001 
* other_state size = 100
* lambda_l2 = 0.0001

### [modelv2](https://github.com/Sshanu/Relation-Classification/blob/master/LCA%20Shortest%20Path/modelv2.ipynb)
* dropout over hidden layer - 0.3

### [modelv3](https://github.com/Sshanu/Relation-Classification/blob/master/LCA%20Shortest%20Path/modelv3.ipynb)
* dropout over word_embedding - 0.3

### [modelv4](https://github.com/Sshanu/Relation-Classification/blob/master/LCA%20Shortest%20Path/modelv4.ipynb)
* dropout over word_embedding - 0.3
* other_state_size = 50

### [modelv5](https://github.com/Sshanu/Relation-Classification/blob/master/LCA%20Shortest%20Path/modelv5.ipynb)
* dropout over word_embedding and hidden_layer - 0.3
* other_state_size = 50
* lambda = 0.00001

### [modelv5](https://github.com/Sshanu/Relation-Classification/blob/master/LCA%20Shortest%20Path/modelv5.ipynb)
* dropout over word_embedding, pos_embedding, dep_embedding of 0.5  
* dropout on hidden_layer of 0.3

``below all models have a learning rate decay at the rate of 0.96 over 2000 steps``

### [modelv6](https://github.com/Sshanu/Relation-Classification/blob/master/LCA%20Shortest%20Path/modelv6.ipynb)
* learning rate decays.

### [modelv7](https://github.com/Sshanu/Relation-Classification/blob/master/LCA%20Shortest%20Path/modelv7.ipynb)
* learnings rate decays.
* dropout on word, pos tags, dep embedding of 0.5
* dropout on hidden layer of 0.3

### [modelv8](https://github.com/Sshanu/Relation-Classification/blob/master/LCA%20Shortest%20Path/modelv8.ipynb)
* learning rate decay
* [word embedding](http://tti-coin.jp/data/wikipedia200.bin) trained over wikipedia
* dropout over hidden layer of 0.3