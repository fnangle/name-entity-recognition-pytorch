# name-entity-recognition-pytorch

NER nlp task with pytorch 用pytorch实现命名实体识别，自己的一点记录

## 数据及评测脚本

### 数据：

CoNLL-2003 dataset and perl Script comes from https://www.clips.uantwerpen.be/conll2003/ner/

如果官方链接挂了，conlleval.pl 见 https://www.clips.uantwerpen.be/conll2000/chunking/conlleval.txt

待评测的数据要注意格式：

> The file conlleval.perl is a Perl program to evaluate the output of sequence labelers producing the B-I-O format. The output of the automatic tagger should be added as a separate column, after the gold-standard annotation.

列和列之间必须为空格（ASCII为20H），只能有一个空格；行之间为换行符\n（ASCII为0AH), 不是回车换行符\r\n（ASCII为0D0AH). 注意使用的是BIO标注法，而不是BIOES标注法。如果你模型的输出是BIOES标注的，需要把E(end)转化为I，S(single)转化为B。

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

```
perl conlleval.pl <test_conlleval.txt >result
```

result:

```
processed 6 tokens with 2 phrases; found: 3 phrases; correct: 1.
accuracy:  66.67%; precision:  33.33%; recall:  50.00%; FB1:  40.00
             MISC: precision:  33.33%; recall:  50.00%; FB1:  40.00  3
```


## Models说明

### Bert only

BertForTokenClassification：详见huggingface https://huggingface.co/transformers/model_doc/bert.html#bertfortokenclassification

需要下载bert-base-cased模型：
https://huggingface.co/bert-base-cased#list-files 

pytorch_model.bin、config.json、vocab.txt 下载三个文件，保存到你的路径中

### Bert+CRF

使用开源的crf包；https://github.com/kmkurn/pytorch-crf/

具体方法见文档 https://pytorch-crf.readthedocs.io/ 和issue29


#### CRF说明

CRF损失函数为：loss=真实路径得分(10)/总路径分数之和(N)，让真实路径得分的比例越来愈高。

分数 = exp(每句话的转移得分+每句话的发射得分)

	- 发射分数：第i个词被标记为第j个标签的预测概率
	- 转移分数：标签j k之间的转移分数
	
每句话的转移得分 = sum_sentence(word_transition_score)

每句话的发射得分 = sum_sentence(word_emission_score)

输出之间的关联仅发生在相邻位置，并且关联是指数加性的。


CRF层能从训练数据中获得约束性的规则

CRF层可以为最后预测的标签添加一些约束来保证预测的标签是合法的。在训练数据训练过程中，这些约束可以通过CRF层自动学习到。
		这些约束可以是：

- I：句子中第一个词总是以标签“B-“ 或 “O”开始，而不是“I-”
- II：标签“B-label1 I-label2 I-label3 I-…”,label1, label2, label3应该属于同一类实体。例-如，“B-Person I-Person” 是合法的序列, 但是“B-Person I-Organization” 是非法标签序列.
- III：标签序列“O I-label” is 非法的.实体标签的首个标签应该是 “B-“ ，而非 “I-“, 换句话说,有效的标签序列应该是“O B-label”。

有了这些约束，标签序列预测中非法序列出现的概率将会大大降低。

### BiLSTM+CRF
（待填坑）

## 文件说明
- process_data.py : 数据处理，将训练和测试文本映射到二维列表，里层是每句话。  

- Bert_ner.py : 主要的程序，用来训练模型. 

- test_bert_ner.py : 测试，把模型预测的结果写到pred.txt ，真实值写到golden.txt，用于后续输入conlleval.pl脚本

- Bert_crf.py 加上了crf层用来训练

## Eval
```bash
cut -d " " -f1 data/test.txt > word.txt
paste -d " " word.txt golden.txt pred.txt > output
perl conlleval.pl < output
```

## Results

| model | F1 score | description                          |
| ----- | -------- | ------------------------------------ |
| Bert  | 65.77    | epoch=5,batchsize=64,AdamW,lr=5e-5,bert-base-uncased|
| Bert  | 81.53    | epoch=4,batchsize=64,AdamW,lr=5e-5,bert-base-cased|
| Bert-crf  |  82.07   | epoch=4,batchsize=64,AdamW,lr=5e-5,bert-base-cased|

