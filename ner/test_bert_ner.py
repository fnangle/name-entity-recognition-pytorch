
import torch
import Bert_ner
from transformers import BertForTokenClassification
if __name__=="__main__":
    params_dir='model/bert_base_model_beta2.pkl'
    path='/data/yanghan/Bert_related/bert_base_uncased/'

    model=BertForTokenClassification.from_pretrained(path,num_labels=9)
    model.load_state_dict(torch.load(params_dir,map_location=torch.device('cpu')))
    res=Bert_ner.generate_pred(model)
    
    print("Finish test!")