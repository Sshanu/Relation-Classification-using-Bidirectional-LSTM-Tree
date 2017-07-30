## Relation Classification using LSTMs on Sequences and Tree Structures

This implementation is based on [End-to-End Relation Extraction using LSTMs
on Sequences and Tree Structures](http://www.aclweb.org/anthology/P/P16/P16-1105.pdf) paper.



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
