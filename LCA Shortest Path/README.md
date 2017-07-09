## Relation Classification using LSTM Networks along Shortest Dependency Paths

Implementation of [Classifying Relations via Long Short Term Memory Networks
along Shortest Dependency Paths](https://pdfs.semanticscholar.org/0f44/366c1e1446cfd51258c68bd1da14fe9c7f10.pdf?_ga=2.136229944.807016038.1498203433-264083776.1497442258)

![LCA Shortest Path](/img/lca.jpg)

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

``learning rate decay rate 0.96 over 2000 steps``

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