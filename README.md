# name-entity-recognition-pytorch
NER nlp task with pytorch
用pytorch实现命名实体识别，自己的一点记录

## 数据及评测脚本

CoNLL-2003 dataset and perl Script comes from https://www.clips.uantwerpen.be/conll2003/ner/ 

or https://www.clips.uantwerpen.be/conll2000/chunking/

如果官方链接挂了，conlleval.pl 见 https://www.clips.uantwerpen.be/conll2000/chunking/conlleval.txt

测试一下 conlleval.pl 脚本:

`

North B-MISC B-MISC
African E-MISC B-MISC
we O O
Grand B-MISC I-MISC
Prix E-MISC E-MISC
we O O

`

`conlleval.pl < dataset.txt`
