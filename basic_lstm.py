import tensorflow as tf
import numpy as np

def lstm_weights(channel, embedding_dim, state_size):
    init_const = tf.zeros([1, state_size])
    with tf.variable_scope(channel):
        W_i = tf.get_variable("W_i",shape=[embedding_dim, state_size] ,initializer=tf.contrib.layers.xavier_initializer())
        U_i = tf.get_variable("U_i",shape=[state_size, state_size] ,initializer=tf.contrib.layers.xavier_initializer())
        b_i = tf.get_variable("b_i", initializer=init_const)

        W_f = tf.get_variable("W_f",shape=[embedding_dim, state_size] ,initializer=tf.contrib.layers.xavier_initializer())
        U_f = tf.get_variable("U_f",shape=[state_size, state_size] ,initializer=tf.contrib.layers.xavier_initializer())
        b_f = tf.get_variable("b_f", initializer=init_const)

        W_o = tf.get_variable("W_o",shape=[embedding_dim, state_size] ,initializer=tf.contrib.layers.xavier_initializer())
        U_o = tf.get_variable("U_o",shape=[state_size, state_size] ,initializer=tf.contrib.layers.xavier_initializer())
        b_o = tf.get_variable("b_o", initializer=init_const)

        W_g = tf.get_variable("W_g",shape=[embedding_dim, state_size] ,initializer=tf.contrib.layers.xavier_initializer())
        U_g = tf.get_variable("U_g",shape=[state_size, state_size] ,initializer=tf.contrib.layers.xavier_initializer())
        b_g = tf.get_variable("b_g", initializer=init_const)


def lstm_cell(channel, input, hidden_state, cell_state):
        with tf.variable_scope(channel, reuse=True):
            W_i = tf.get_variable("W_i")
            U_i = tf.get_variable("U_i")
            b_i = tf.get_variable("b_i")

            W_f = tf.get_variable("W_f")
            U_f = tf.get_variable("U_f")
            b_f = tf.get_variable("b_f")

            W_o = tf.get_variable("W_o")
            U_o = tf.get_variable("U_o")
            b_o = tf.get_variable("b_o")

            W_g = tf.get_variable("W_g")
            U_g = tf.get_variable("U_g")
            b_g = tf.get_variable("b_g")

            input_gate = tf.sigmoid(tf.matmul(input, W_i) + tf.matmul(hidden_state, U_i) + b_i)
            forget_gate = tf.sigmoid(tf.matmul(input, W_f) + tf.matmul(hidden_state, U_f) + b_f)
            output_gate = tf.sigmoid(tf.matmul(input, W_o) + tf.matmul(hidden_state, U_o) + b_o)
            gt = tf.tanh(tf.matmul(input, W_g) + tf.matmul(hidden_state, U_g) + b_g)
            cell_state = input_gate * gt + forget_gate * cell_state
            hidden_state = output_gate * tf.tanh(cell_state)
            return hidden_state, cell_state
