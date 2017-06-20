import sys, _pickle as pickle
f = open('train_data.pkl', 'rb')
data, relation, e1, e2, e1_pos, e2_pos = pickle.load(f)
