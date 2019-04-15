# Relation Classification 

[![MIT License](https://img.shields.io/badge/license-MIT-green.svg)](https://opensource.org/licenses/MIT) 

Relation classification aims to categorize into predefined classes the relations btw pairs of given entities in texts. There are two ways to represent relations between entities using: recurrent neural networks (RNNs) and convolutional neural networks (CNNs). We have implemented three LSTM-RNN architectures for solving the task of relation classification:
* [Relation classification using LSTM Networks along Shortest Dependency Paths.](https://github.com/Sshanu/Relation-Classification/tree/master/LCA%20Shortest%20Path)
* [Relation classification using bidirectional LSTM Networks on LCA Sub Tree.](https://github.com/Sshanu/Relation-Classification/tree/master/LCA%20SubTree)
* [Relation classification using LSTMS on Sequences and Tree Structures.](https://github.com/Sshanu/Relation-Classification/tree/master/LSTM%20Seq%20and%20Tree)

We achieve better performance for solving this task using the last approach "[Relation classification using LSTMS on Sequences and Tree Structures.](https://github.com/Sshanu/Relation-Classification/tree/master/LSTM%20Seq%20and%20Tree)".


### References:

> **End-to-End Relation Extraction using LSTMs on Sequences and Tree Structures**<br>
> Makoto Miwa, Mohit Bansal<br>
> [http://www.aclweb.org/anthology/P/P16/P16-1105.pdf](http://www.aclweb.org/anthology/P/P16/P16-1105.pdf)
> 
> **Abstract:** *We present a novel end-to-end neural
model to extract entities and relations between them. Our recurrent neural network based model captures both word sequence and dependency tree substructure
information by stacking bidirectional treestructured LSTM-RNNs on bidirectional
sequential LSTM-RNNs. This allows our
model to jointly represent both entities and
relations with shared parameters in a single model. We further encourage detection of entities during training and use of
entity information in relation extraction
via entity pretraining and scheduled sampling. Our model improves over the stateof-the-art feature-based model on end-toend relation extraction, achieving 12.1%
and 5.7% relative error reductions in F1-
score on ACE2005 and ACE2004, respectively. We also show that our LSTMRNN based model compares favorably to
the state-of-the-art CNN based model (in
F1-score) on nominal relation classification (SemEval-2010 Task 8). Finally, we
present an extensive ablation analysis of
several model components*

> **Classifying Relations via Long Short Term Memory Networks along Shortest Dependency Paths**<br>
> Yan Xu, Lili Mou, Ge Li, Yunchuan Chen, Hao Peng, Zhi Jin<br>
> [http://www.emnlp2015.org/proceedings/EMNLP/pdf/EMNLP206.pdf](http://www.emnlp2015.org/proceedings/EMNLP/pdf/EMNLP206.pdf)
> 
> **Abstract:** *Relation classification is an important research arena in the field of natural language processing (NLP). In this paper, we
present SDP-LSTM, a novel neural network to classify the relation of two entities in a sentence. Our neural architecture
leverages the shortest dependency path
(SDP) between two entities; multichannel recurrent neural networks, with long
short term memory (LSTM) units, pick
up heterogeneous information along the
SDP. Our proposed model has several distinct features: (1) The shortest dependency
paths retain most relevant information (to
relation classification), while eliminating
irrelevant words in the sentence. (2) The
multichannel LSTM networks allow effective information integration from heterogeneous sources over the dependency
paths. (3) A customized dropout strategy
regularizes the neural network to alleviate overfitting. We test our model on the
SemEval 2010 relation classification task,
and achieve an F1-score of 83.7%, higher
than competing methods in the literature.*
