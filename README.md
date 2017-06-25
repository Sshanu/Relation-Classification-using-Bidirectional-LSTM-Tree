1. Prepocessing the training and test data SemEval2010.
2. Extract the two entities from training and test data.
3. Parse the data for Dependency Tree and Types, POS Tags.
4. Extract the shortest path in Dependency tree for the entities.
5. Build the embedding layer for Word, POS Tags, Dependency Types.
6. LSTM Network for SDP.
7. Pool the Lstms from all the nodes.
8. Linear Layer with regularization and SoftMax for classification.

## modelv1 
## modelv2 
dropout over hidden layer 0.3