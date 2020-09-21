# name-entity-recognition-pytorch
NER nlp task with pytorch
用pytorch实现命名实体识别，自己的一点记录

## 数据及评测脚本

CoNLL-2003 dataset and perl Script comes from https://www.clips.uantwerpen.be/conll2003/ner/ 

or https://www.clips.uantwerpen.be/conll2000/chunking/

如果官方链接挂了，conlleval.pl 见 https://www.clips.uantwerpen.be/conll2000/chunking/conlleval.txt

### 测试一下 conlleval.pl 脚本:

test_conlleval.txt:

```  
North B-MISC B-MISC
African E-MISC B-MISC
we O O
Grand B-MISC I-MISC
Prix E-MISC E-MISC
we O O

```
cmd:

`perl conlleval.pl <test_conlleval.txt >res`

result:

processed 6 tokens with 2 phrases; found: 3 phrases; correct: 1.
accuracy:  66.67%; precision:  33.33%; recall:  50.00%; FB1:  40.00
             MISC: precision:  33.33%; recall:  50.00%; FB1:  40.00  3
