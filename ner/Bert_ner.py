import process_data
from torch.utils.data import TensorDataset,DataLoader
import torch
import logging
import numpy as np
from transformers import BertTokenizer ,BertForTokenClassification ,AdamW

MAXLEN = 128 -2
BATCHSIZE = 64

path="/data/yanghan/Bert_related/bert_base_cased/"
config_dir=path
tokenizer=BertTokenizer.from_pretrained(path)
model=BertForTokenClassification.from_pretrained(path)


#能把文本列表转为token列表，pad，truncate，有cls和sep
def convert_text_to_ids(tokenizer, sentence, limit_size=MAXLEN):
    t = sentence[:limit_size]
    encoded_ids = tokenizer.encode(t)
    if len(encoded_ids) < limit_size + 2:
        tmp = [0] * (limit_size + 2 - len(encoded_ids))
        encoded_ids.extend(tmp)
    return encoded_ids

#label应该与input_ids对应 第一位补0 对应【CLS】
def convert_label_to_ids(sentence, limit_size=MAXLEN):
    labels=[0]
    if len(sentence) < limit_size +2:
        tmp = [0] * (limit_size +2 - len(sentence)-1)
        labels.extend(sentence)
        labels.extend(tmp)
    return labels

input_ids=[convert_text_to_ids(tokenizer,sen) for sen in process_data.train_samples]
input_labels=[convert_label_to_ids(seq) for seq in process_data.train_labels]
# print(input_labels[:3],len(input_labels[0]))

def get_att_masks(input_ids):
    atten_masks=[]
    for seq in input_ids:
        seq_mask=[float(i)>0 for i in seq]
        atten_masks.append(seq_mask)
    return atten_masks

atten_token_train = get_att_masks(input_ids)

'''构建数据集和数据迭代器，设定 batch_size 大小'''
train_set=TensorDataset(torch.LongTensor(input_ids),torch.LongTensor(atten_token_train),
                        torch.Tensor(input_labels)) #list只能用torch.Tensor转换
train_loader=DataLoader(train_set,
                        batch_size=BATCHSIZE,
                        shuffle=True)
# for i, (train, mask, label) in enumerate(train_loader):
#     print(train.shape, mask.shape, label.shape)
    # break


input_ids2=[convert_text_to_ids(tokenizer,sen) for sen in process_data.test_samples]
input_labels2=[convert_label_to_ids(seq) for seq in process_data.test_labels]
atten_tokens_eval = get_att_masks(input_ids2)
test_set=TensorDataset(torch.LongTensor(input_ids2),torch.LongTensor(atten_tokens_eval),
                       torch.Tensor(input_labels2))
test_loader=DataLoader(test_set,
                       batch_size=BATCHSIZE,
                       shuffle=False)


device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_model(net,epoch=4):
    avg_loss=[]
    net.train()
    net.to(device)
    optimizer = AdamW(net.parameters(), lr=5e-5)
    accumulation_steps=1
    for e in range(epoch):
        for batch_idx,(data, mask,target) in enumerate(train_loader):
            data,mask,target=data.to(device),mask.to(device),target.long().to(device)
            loss,logits = net(data,attention_mask=mask,labels=target)

            loss = loss / accumulation_steps  # 梯度积累
            avg_loss.append(loss.item())
            loss.backward()
            
            if batch_idx% accumulation_steps==0:
                optimizer.step()
                optimizer.zero_grad()
            
            if batch_idx % 5 == 0:
                logging.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss:{:.6f}'.format(
                    e + 1, batch_idx, len(train_loader), 100. *
                    batch_idx / len(train_loader), np.array(avg_loss).mean()
                ))
    print("Finish training!")
    return net

def predict(logits):
    res=torch.argmax(logits,dim=2) # 按行取每行最大的列下标
    return res

def generate_pred(net):
    net.eval()
    net = net.to(device)
    dir='pred.txt'
    dir2='golden.txt'
    fw=open(dir,'w')
    fw2=open(dir2,'w')
    label2tag={v:k for k,v in process_data.tag2label.items()}
    total=0
    with torch.no_grad():
        for batch_idx, (data, mask, label) in enumerate(test_loader):
            data, mask, label = data.to(device), mask.to(device), label.to(device)
            logits = net(data, token_type_ids=None, attention_mask=mask)  # 调用model模型时不传入label值。
            ## logits的形式为（元组类型，第0个元素是各类别概率）
            # print(logits[0].size(),label.shape) ;break
            #torch.Size([8, 128, 9]) torch.Size([8, 128])
            
            pred=predict(logits[0]) #torch.Size([8, 128])
            # print(pred);break

            for i,sen in enumerate(data):
                for j,word in enumerate(sen):
                    word=tokenizer.convert_ids_to_tokens(word.item())
                    if word =='[SEP]':
                        
                        end=j
                        for k in range(1,end):
                            total+=1
                            tmp=pred[i][k].item(); 
                            predtag=label2tag.get(tmp) 
                            fw.write(str(predtag)+"\n")
                            goldentag=label2tag.get(label[i][k].item())
                            fw2.write(str(goldentag)+'\n')
                fw.write("\n")
                total+=1
                fw2.write('\n')
        fw.close()
        fw2.close()
        print(total)

if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s:%(levelname)s: %(message)s', level=logging.INFO)
    # fn_model = fn_class()
    pre_net = BertForTokenClassification.from_pretrained(path,num_labels=9)
    params_dir = 'model/bert_base_model_beta2.pkl'

    model = train_model(pre_net)
    torch.save(model.state_dict(), params_dir)  # 保存模型参数
