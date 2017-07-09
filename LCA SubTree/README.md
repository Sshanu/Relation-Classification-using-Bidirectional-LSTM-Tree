## Relation Classification using LSTMs on LCA Sub Tree

LSTMs are applied on the Sub Tree of Lowest Ancestor of two entities as a sequence when traversed.

Model | Train-Accuracy | Test-Accuracy| Epochs
--- | --- | ---| ---
model2v1 | ? | 54.6 | 11
model2v2 | ? | 55.2 | 10



* dropout on hidden layer of 0.3
* Learning rate = 0.001 
* Learning rate decay = 0.96
* state size = 100
* lambda_l2 = 0.0001


### [model2v1](https://github.com/Sshanu/Relation-Classification/blob/master/LCA%20SubTree/model2v1.ipynb)
* Foward LSTM on the sequnence traversed on LCA Sub Tree.

### [model2v2](https://github.com/Sshanu/Relation-Classification/blob/master/LCA%20SubTree/model2v2.ipynb)
* Bidirectional LSTM on the sequnence traversed on LCA Sub Tree.
