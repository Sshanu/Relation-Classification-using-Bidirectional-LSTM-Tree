1. Prepocessing the training and test data SemEval2010.
2. Extract the two entities from training and test data.
3. Parse the data for Dependency Tree and Types, POS Tags.
4. Extract the shortest path in Dependency tree for the entities.
5. Build the embedding layer for Word, POS Tags, Dependency Types.
6. LSTM Network for SDP.
7. Pool the Lstms from all the nodes.
8. Linear Layer with regularization and SoftMax for classification.



Model | Train-Accuracy | Test-Accuracy| Epochs
--- | --- | ---| ---
modelv1 | 99.45 | 62 | 10
modelv2 | ? | ? | 10
modelv3 | 99.45 | 62 | 20
modelv4 | 83 | 60 | 20
modelv5 | 78 | 57 | 20


### modelv1 
* No dropout 
* Learning rate = 0.001 
* other_state size = 100
* lambda_l2 = 0.0001

### modelv2 
* dropout over hidden layer - 0.3

### modelv3
* dropout over word_embedding - 0.3

### modelv4
* dropout over word_embedding - 0.3
* other_state_size = 50

### modelv5
* dropout over word_embedding and hidden_layer - 0.3
* other_state_size = 50
* lambda = 0.00001