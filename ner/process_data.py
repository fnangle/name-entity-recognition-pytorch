import os
import pickle

tag2label={ "O":0,
            "B-PER": 1, "I-PER": 2,
            "B-LOC": 3, "I-LOC": 4,
            "B-ORG": 5, "I-ORG": 6,
            "B-MISC":7, "I-MISC":8}

train_samples=[]
train_labels=[]

conll_dir='./data'
train_dir= os.path.join(conll_dir,'train.txt')
test_dir= os.path.join(conll_dir,'test.txt')

#读取文件，转为文件列表, 要先给train和test的文件末尾加个换行 否则最后一句话会漏掉
def file_list(f_dir):
    texts=[];labels=[]
    with open(f_dir) as fo :
        lines=fo.readlines()
        sen, label = [], []
        for l in lines:    
            if l is not "\n": 
                word=str(l).split(" ")[0]
                tag=str(l).strip().split(" ")[3]
                sen.append(word)
                label.append(tag2label.get(tag))
            elif l is '\n' :   
                texts.append(sen)
                labels.append(label)
                sen, label = [],[]
    return texts, labels

train_samples,train_labels = file_list(train_dir)
test_samples,test_labels = file_list(test_dir)
print(test_samples[-2:])
print(test_labels[-2:])
# print(train_samples[-2:])
# print(train_labels[-2:])


"""
[['-DOCSTART-'], ['EU', 'rejects', 'German', 'call', 'to', 'boycott', 'British', 'lamb', '.']]
[['O'], ['B-ORG', 'O', 'B-MISC', 'O', 'O', 'O', 'B-MISC', 'O', 'O']]
[[0], [5, 0, 7, 0, 0, 0, 7, 0, 0]]
"""


