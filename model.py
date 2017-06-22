import sys, os, _pickle as pickle
import tensorflow as tf
import numpy as np
from tensorflow.contrib import learn

data_dir = 'data'
ckpt_dir = 'checkpoint'
word_embd_dir = 'checkpoint/word_embd'
model_dir = 'checkpoint/model'
f = open('train_data.pkl', 'rb')
data, relation, e1, e2, e1_pos, e2_pos = pickle.load(f)

max_document_length = max([len(x.split(" ")) for x in data])
word_embd_dim = 200
pos_embd_dim = 25
dep_embd_dim = 25
vocab_size = 400000
pos_size = 10
relation_classes = 19
vocab_processor = learn.preprocessing.VocabularyProcessor.restore(os.path.join(data_dir, "word_vocab"))

# vocab_processor.save(os.path.join(data_dir, "word_vocab"))
with tf.name_scope("input"):
    word = tf.placeholder(tf.int32, [None, None], name="word")
    pos_tag = tf.placeholder(tf.int32, [None, None], name="pos_tag")
    relation = tf.placeholder(tf.int32, [None,relation_classes], name="relation")
with tf.name_scope("word_embedding"):
    W = tf.Variable(tf.constant(0.0, shape=[vocab_size, word_embd_dim]), name="W")
    embedding_placeholder = tf.placeholder(tf.float32,[vocab_size, word_embd_dim])
    embedding_init = W.assign(embedding_placeholder)
    embedded_word = tf.nn.embedding_lookup(W, word)
    word_embedding_saver = tf.train.Saver({"word_embedding/W": W})

with tf.name_scope("pos_embedding"):
    W = tf.Variable(tf.random_uniform([pos_size, pos_embd_dim], name="W"))
    embedded_pos = tf.nn.embedding_lookup(W, pos_tag)
    pos_embedding_saver = tf.train.Saver({"pos_embedding/W": W})

sess  =tf.Session()
sess.run(tf.global_variables_initializer())
# sess.run(embedding_init, feed_dict={embedding_placeholder: word_embd})
# word_embedding_saver.save(sess, word_embd_dir + '/word_embd')
latest_embd = tf.train.latest_checkpoint(word_embd_dir)
word_embedding_saver.restore(sess, latest_embd)
input_word = [","]
x = np.array(list(vocab_processor.transform(input_word)))
x
iin = sess.run(embedded_word, {word:x})
x = np.zeros((1,1))
iin[0][0]
x
np.shape(iin)
