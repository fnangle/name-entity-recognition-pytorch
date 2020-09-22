# name-entity-recognition-pytorch
NER nlp task with pytorch
用pytorch实现命名实体识别，自己的一点记录

## 数据及评测脚本
  
### 数据：

CoNLL-2003 dataset and perl Script comes from https://www.clips.uantwerpen.be/conll2003/ner/ 

如果官方链接挂了，conlleval.pl 见 https://www.clips.uantwerpen.be/conll2000/chunking/conlleval.txt

待评测的数据要注意格式：

> The file conlleval.perl is a Perl program to evaluate the output of sequence labelers producing the B-I-O format. The output of the automatic tagger should be added as a separate column, after the gold-standard annotation.
  
  列和列之间必须为空格（ASCII为20H），只能有一个空格；行之间为换行符\n（ASCII为0AH),
不是回车换行符\r\n（ASCII为0D0AH). 
  注意使用的是BIO标注法，而不是BIOES标注法。如果你模型的输出是BIOES标注的，需要把E(end)转化为I，S(single)转化为B。

ps:自己在处理之前给train.txt test.txt 文件末尾各加了一个空行

### 测试一下 conlleval.pl 脚本:

test_conlleval.txt （最后一列是预测的结果，在golden值的后一列）:

```  
North B-MISC B-MISC
African E-MISC B-MISC
we O O
Grand B-MISC I-MISC
Prix E-MISC E-MISC
we O O
```
cmd:

`perl conlleval.pl <test_conlleval.txt >result`

result:
```
processed 6 tokens with 2 phrases; found: 3 phrases; correct: 1.
accuracy:  66.67%; precision:  33.33%; recall:  50.00%; FB1:  40.00
             MISC: precision:  33.33%; recall:  50.00%; FB1:  40.00  3
```

## results
|model|F1 score|description|
|:-:|:-:|:-:|
|Bert|65.77|epoch=5，batchsize=64,AdamW ,lr=5e-5|

## models

#### Bert only

BertForTokenClassification：详见hugging face https://huggingface.co/transformers/model_doc/bert.html#bertfortokenclassification

#### Bert+CRF

（待填坑。。。）
