import sys, os, _pickle as pickle
import tensorflow as tf
import numpy as np
from tensorflow.contrib import learn
data_dir = 'data'
ckpt_dir = 'checkpoint'
word_embd_dir = 'checkpoint/word_embd'
f = open('train_data.pkl', 'rb')
data, relation, e1, e2, e1_pos, e2_pos = pickle.load(f)
max_document_length = max([len(x.split(" ")) for x in data])
max_document_length = 97
word_embd_dim = 200
pos_embd_dim = 25
dep_embd_dim = 25
vocab_size = 400000
vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
vocab = []
embd = []
file = open('glove.6B.200d.txt', 'r')
for line in file.readlines():
    row = line.strip().split()
    vocab.append(row[0])
    embd.append(row[1:])
file.close()
pretrain = vocab_processor.fit(vocab)
vocab_processor.save(os.path.join(data_dir, "word_vocab"))
word_embd = np.asarray(embd)

with tf.name_scope("word_embedding"):
    W = tf.Variable(tf.constant(0.0, shape=[vocab_size, word_embd_dim]), name="W")
    embedding_placeholder = tf.placeholder(tf.float32,[vocab_size, word_embd_dim])
    embedding_init = W.assign(embedding_placeholder)
    word_embedding_saver = tf.train.Saver({"word_embedding/W": W})

sess  =tf.Session()
sess.run(tf.global_variables_initializer())
sess.run(embedding_init, feed_dict={embedding_placeholder: word_embd})
word_embedding_saver.save(sess, word_embd_dir + '/word_embd')
