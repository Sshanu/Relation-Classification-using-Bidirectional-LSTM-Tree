import re, sys, nltk
import _pickle as pickle
lines = []
for line in open("TRAIN_FILE.TXT"):
    m = re.match(r'^([0-9]+)\s"(.+)"$', line.strip())
    if(m is not None):
        lines.append(m.group(2))
data = []
e1 = []
e2 = []
e1_pos = []
e2_pos = []
for line in lines:
    text = []
    t = line.split("<e1>")
    text.append(t[0])
    t = t[1].split("</e1>")
    e1_text = text
    e1_text = " ".join(e1_text)
    e1_text = nltk.word_tokenize(e1_text)
    e1_pos.append(len(e1_text))
    text.append(t[0])
    e1.append(t[0])
    t = t[1].split("<e2>")
    text.append(t[0])
    t = t[1].split("</e2>")
    e2.append(t[0])
    e2_text = text
    e2_text = " ".join(e2_text)
    e2_text = nltk.word_tokenize(e2_text)
    e2_pos.append(len(e2_text))
    text.append(t[0])
    text.append(t[1])
    text = " ".join(text)
    text = nltk.word_tokenize(text)
    text = " ".join(text)
    data.append(text)


file = open("relation.txt")
relation = []
for line in file:
    relation.append(line.split()[1])
file.close()

with open('train_data.pkl', 'wb') as f:
    pickle.dump((data, relation, e1, e2, e1_pos, e2_pos), f)
    f.close()
