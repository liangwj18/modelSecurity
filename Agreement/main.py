import pandas as pd
from sklearn import model_selection
from model import AgreementModel
from tqdm import tqdm
import os

import argparse 

import torch
import torch.nn as nn
import torch.utils.data as data_utl
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import BertTokenizer
import json
import shutil

from datasets import DatasetDict

class AgreementDataset(data_utl.Dataset):
    def __init__(self, data):
        self.data = data
    def __getitem__(self, index):
        return self.data[index]
    def __len__(self):
        return len(self.data)
def collate_fn(batch):
    """
    对batch数据进行处理
    :param batch: [一个getitem的结果，getitem的结果,getitem的结果]
    :return: 元组
    """
    labels, sentences, input_ids, attention_mask, sentiment = zip(*batch)
    # sentence_vectors['attention_mask'] = torch.LongTensor([config.ws.transform(i,max_len=config.max_len) for i in reviews])
    labels = torch.LongTensor(labels)
    # for input_id in input_ids:
    #     print(len(input_id))
    input_ids = torch.LongTensor(input_ids)
    attention_mask = torch.LongTensor(attention_mask)
    sentiment = torch.tensor(sentiment)

    return labels, sentences, {"input_ids":input_ids, 'attention_mask':attention_mask}, sentiment

def Handle_OriginData(args):

    if args.roberta:
        tokenizer = BertTokenizer.from_pretrained("hfl/chinese-roberta-wwm-ext")
    else:
        tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    # print(tokenizer(" "))
    # assert 1 == 2
    sum_len = 0
    f = open(args.data_path, 'r')
    # print(args.data_path)
    lines = f.readlines()
    # print(lines)

    if args.model_with_sentiment:
        pathh = args.data_path.split(".csv")[0]+"_sentiment.json"
        if not os.path.exists(pathh):
            my_sentiment(lines,pathh,args)
        sentiments = json.load(open(pathh,'r'))
    
    Datasets = []
    label_count = [0, 0, 0] if args.label2 == 1 else [0, 0]
    last_sentence = ""
    for line_index in tqdm(range(0, len(lines))):
        line = lines[line_index]
        tmp = line.split("\n")[0].split("\t")
        # print(tmp)
        sentence = tmp[0]
        if len(sentence.split("[SEP]"))<2:
            continue
        if len(sentence) > 80: 
            continue
        if args.use_only_answer == 1:
            # print(sentence)
            sentence = sentence.split("[SEP]")[1]
        if len(tmp)>=2:
            label = int(tmp[1])
        else:
            label = -1

        if line_index > 0 and sentence == last_sentence:
            continue
        last_sentence = sentence
        if label == 3:
            continue
        if label == 2 and args.label2 == 0:
            label = 0
        label_count[label] += 1

        sentence = sentence[:args.sentence_max_length-2]
        for i in range(len(sentence), args.sentence_max_length-2):
            sentence += ' '
        # print(len(sentence))
        tmp = tokenizer(sentence)
        if (line_index == 0):
            print(tmp)
        y = len(tmp['attention_mask'])
        tmp['attention_mask'] += [0 for i in range(y, args.sentence_max_length)]
        tmp['input_ids'] += [tmp['input_ids'][y-1] for i in range(y, args.sentence_max_length)]

        sentiment = sentiments[line_index] if args.model_with_sentiment else [0,0,0,0,0]
        Datasets.append([label,  sentence, tmp['input_ids'], tmp['attention_mask'], sentiment])
        # print(Datasets[-1][2].shape)

    print(Datasets[0])
    if args.generate == 1:
        Train_Dataset = AgreementDataset(Datasets)
        return [DataLoader(Train_Dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True, collate_fn = collate_fn),
                None,None]
    else:
        if args.data_path == "./data/round4_train.csv":
            print("round4")
            Train_data, Test_data = model_selection.train_test_split(Datasets[:1492], test_size=0.15,random_state=0)            
            for k in range(1492, len(Datasets)-1, 2):
                Train_data.append(Datasets[k])
                Test_data.append(Datasets[k+1])
        else:
            Train_data, Test_data = model_selection.train_test_split(Datasets, test_size=0.15,random_state=0)        
        Train_Dataset = AgreementDataset(Train_data)
        Test_Dataset = AgreementDataset(Test_data)
        print("Train_length = {} Eval_length = {}",len(Train_Dataset), len(Test_Dataset))

        dataset = {
            'train': Train_Dataset,
            'eval': Test_Dataset,
        }
        # print(dataset)
        return [DataLoader(Train_Dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, collate_fn = collate_fn),
                DataLoader(Test_Dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, collate_fn = collate_fn),
                label_count]

def calc_weight(label_count, args):
    label_count = torch.tensor([1, 1, 1])/torch.tensor(label_count) if args.label2 == 1 else torch.tensor([1, 1])/torch.tensor(label_count)
    sample_num = torch.sum(label_count)
    label_count = (label_count/torch.tensor([sample_num,sample_num, sample_num])).to(args.device) if args.label2 == 1 else (label_count/torch.tensor([sample_num,sample_num])).to(args.device)
    return label_count

def test(model, Eval_Dataloader, args):
    model.eval()
    sum_acc = 0
    sum_sample = 0
    # sum_not_one = 0
    for batch_i, batch in tqdm(enumerate(Eval_Dataloader)):
        batch_labels,  batch_sentences,batch_x, sentiment= batch
        batch_labels = batch_labels.to(args.device)
        batch_x = {key:value.to(args.device) for key, value in batch_x.items()}
        predict_distributions = model(batch_x,sentiment.to(args.device))

        predict_labels = torch.argmax(predict_distributions, dim = 1)
        now_acc = torch.sum(predict_labels == batch_labels)
        sum_acc += now_acc
        sum_sample += predict_labels.shape[0]
        # sum_not_one += torch.sum(predict_labels != 1)
    return sum_acc / sum_sample


def train(args, Train_Dataloader, label_count, Eval_Dataloader):
  
    # model = LSTM_Sentiment(args).to(args.device)
    model = AgreementModel(3, args.dropout, args.model_with_sentiment, args.roberta, args).to(args.device) if args.label2 == 1 else AgreementModel(2).to(args.device)
    # optimizer = torch.optim.Adam(model.parameters(), lr = args.learning_rate)
    # optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
    # optimizer = torch.optim.Adagrad(model.parameters(), lr=1e-5, lr_decay=0, weight_decay=0, initial_accumulator_value=0, eps=1e-10)
    optimizer = torch.optim.RMSprop(model.parameters(), lr = args.learning_rate)


    label_weight = calc_weight(label_count, args)
    print("label_weight:", label_weight)
    criterion = nn.CrossEntropyLoss(weight = label_weight)

    print(args.resume_checkpoint_path)
    resume_model = "origin" if args.resume == 0 else args.resume_checkpoint_path.split("runs/")[1].split("gpu166")[0]
    writer = SummaryWriter(comment = "{}_LR_{}_Batch_{}_onlyAnswer_{}_Label2_{}_Resume_{}_Sentiment_{}_Model_{}".format(args.data_path.split("/")[-1].split(".")[0],args.learning_rate, args.batch_size, args.use_only_answer, args.label2, resume_model, args.model_with_sentiment, args.base_model))
    logdir = writer.log_dir
    args.save_checkpoint_path = logdir
    json.dump(args.__dict__, open(os.path.join(logdir, 'args.json'), 'w'))
    
    start_epoch = 0
    best_acc = 0
    if args.resume == 1: # resume为参数，第一次训练时设为0，中断再训练时设为1
        model_path = os.path.join(args.resume_checkpoint_path, 'best_checkpoint.pth.tar')
        assert os.path.isfile(model_path)
        checkpoint = torch.load(model_path,map_location={'cuda:3':args.device})
        best_acc = checkpoint['best_acc'].to(args.device)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print('Load checkpoint at epoch {}.'.format(start_epoch))
        # print('Best accuracy so far {}.'.format(best_acc))

    for epoch in range(args.epochs):
        model.train()
        sum_acc = 0
        sum_sample = 0
        sum_not_one = 0
        for batch_i, batch in tqdm(enumerate(Train_Dataloader)):
            batch_labels,  batch_sentences, batch_x,sentiment = batch
            batch_labels = batch_labels.to(args.device)
            batch_x = {key:value.to(args.device) for key, value in batch_x.items()}
            predict_distributions = model(batch_x,sentiment.to(args.device))

            # negation_loss *= args.alpha
            # sentiment_loss *= args.alpha
            
            # print(predict_labels)
            loss = criterion(predict_distributions, batch_labels)
            predict_labels = torch.argmax(predict_distributions, dim = 1)
            now_acc = torch.sum(predict_labels == batch_labels)
            sum_acc += now_acc
            sum_sample += predict_labels.shape[0]
            # sum_not_one += torch.sum(predict_labels != 1)
            # label_loss = criterion(predict_distributions, batch_labels)
            # loss = label_loss + negation_loss + sentiment_loss
            # loss = label_loss
            # print(loss)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            n_iter = epoch*len(Train_Dataloader) + batch_i
            writer.add_scalar('Loss/train', loss.item(), n_iter)
            writer.add_scalar('Accuracy/train', now_acc/predict_labels.shape[0], n_iter)
            # writer.add_scalar('sentimentLoss/train', sentiment_loss.item(), n_iter)
            # writer.add_scalar('negationLoss/train', negation_loss.item(), n_iter)
            # writer.add_scalar('labelLoss/train', label_loss.item(), n_iter)

      
        print('Epoch: {}, Step: {}/{}, Loss: {}, Acc: {}'.format(epoch, 
        batch_i, len(Train_Dataloader), loss.item(), sum_acc/sum_sample))
        # print('predict_labels:', predict_labels)
        # print('sum_not_one:', sum_not_one)
        # sum_not_one = 0
        current_acc = sum_acc / sum_sample
        
        sum_acc = 0
        sum_sample = 0

        eval_acc = test(model, Eval_Dataloader, args)
        print("Eval acc",eval_acc)
        writer.add_scalar('Accuracy/eval', eval_acc, epoch)
        is_best = eval_acc > best_acc
        best_acc = max(eval_acc, best_acc)
        checkpoint = {
            'best_acc': best_acc,
            'epoch': epoch + 1,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }
        model_path = os.path.join(args.save_checkpoint_path, 'checkpoint.pth.tar')
        best_model_path = os.path.join(args.save_checkpoint_path, 'best_checkpoint.pth.tar')
        torch.save(checkpoint, model_path)
        if is_best:
            shutil.copy(model_path, best_model_path)

from transformers import pipeline

def test_generate(args, Train_Dataloader):
    
    model = AgreementModel(3, args.dropout, args.model_with_sentiment, args.roberta, args).to(args.device) if args.label2 == 1 else AgreementModel(2).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr = args.learning_rate)

    start_epoch = 0
    best_acc = 0
    if args.resume == 1: # resume为参数，第一次训练时设为0，中断再训练时设为1
        model_path = os.path.join(args.resume_checkpoint_path, 'best_checkpoint.pth.tar')
        print(model_path)
        assert os.path.isfile(model_path)
        checkpoint = torch.load(model_path,map_location={'cuda:3':args.device})
        best_acc = checkpoint['best_acc'].to(args.device)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print('Load checkpoint at epoch {}.'.format(start_epoch))
        print('Best accuracy so far {}.'.format(best_acc))

    model.eval()
    output_path = args.data_path.split(".csv")[0]+"_out.csv"
    sum_acc = 0
    sum_sample = 0
    actural_predict_sample = [[],[],[]]
    for i in range(3):
        actural_predict_sample[i].append(0)
        actural_predict_sample[i].append(0)
        actural_predict_sample[i].append(0)
    with open(output_path, 'w', encoding = 'utf-8') as f:
        for batch_i, batch in tqdm(enumerate(Train_Dataloader)):
            batch_labels,  batch_sentences, batch_x,sentiment = batch
            batch_labels = batch_labels.to(args.device)
            batch_x = {key:value.to(args.device) for key, value in batch_x.items()}
            predict_distributions = model(batch_x,sentiment.to(args.device))
            predict_labels = torch.argmax(predict_distributions, dim = 1)
            sum_acc += sum(predict_labels == batch_labels)
            sum_sample += predict_labels.shape[0]

            for k in range(len(batch_sentences)):
                f.write(batch_sentences[k].split(" ")[0]+" "+str(batch_labels[k].cpu().detach().numpy())+" "+str(predict_labels[k].cpu().detach().numpy())+" "+str(predict_distributions[k][predict_labels[k]].cpu().detach().numpy())+"\n")
                xxx = batch_labels[k].cpu().detach().numpy()
                if xxx == -1:
                    xxx=predict_labels[k].cpu().detach().numpy()
                actural_predict_sample[xxx][predict_labels[k].cpu().detach().numpy()] +=1
    print("class = {} {} {}".format(str(actural_predict_sample[0][0]/sum_sample), str(actural_predict_sample[1][1]/sum_sample), str(actural_predict_sample[2][2]/sum_sample)))
    f = open("output_gen.txt",'a')
    f.write("class = {} {} {}\n".format(str(actural_predict_sample[0][0]/sum_sample),str(actural_predict_sample[1][1]/sum_sample),str(actural_predict_sample[2][2]/sum_sample)))
    f.close()
    print("acc:{}\n".format(str(sum_acc/sum_sample)))
    tps = 0
    fps_add_tps = 0
    sample = [0, 0 ,0]
    for i in range(3):
        print("class {}:------".format(i))
        tp = actural_predict_sample[i][i]
        fp_add_tp = (actural_predict_sample[0][i] + actural_predict_sample[1][i] + actural_predict_sample[2][i])
  
        tps+=tp
        fps_add_tps+=fp_add_tp
        fn_add_tp = (actural_predict_sample[i][0] + actural_predict_sample[i][1] + actural_predict_sample[i][2])
        sample[i] = fn_add_tp
        print("precision = {}".format(str(tp/fp_add_tp)))
        print("Recall = {}".format(str(tp/fn_add_tp)))
    print("micro-average = {}".format(str(tps/fps_add_tps)))
    sums = sum(sample)
    print("sample = {} {} {}".format(str(sample[0]/sums), str(sample[1]/sums), str(sample[2]/sums)))
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_only_answer', type = int, default = 0)
    parser.add_argument('--label2', type = int, default = 1)
    parser.add_argument('--data_path', type=str, default='./data/round4_test.csv')
    parser.add_argument('--generate', type = int, default = 1)
    # parser.add_argument('--data_vector_path', type=str, default='./weibo.json')
    parser.add_argument('--batch_size', type = int, default = 16)
    parser.add_argument('--num_workers', type = int, default = 0)
    parser.add_argument('--sentence_max_length', type = int, default = 60)
    # parser.add_argument('--sentiment_dict_path', type = str, default = '../data/Chinese_Corpus-master/sentiment_dict/ntusd/')
    # parser.add_argument('--using_Transformer', action = "store_true")
    parser.add_argument('--learning_rate', type = float, default = 1e-5)
    parser.add_argument('--epochs', type = int, default = 40)
    parser.add_argument('--device', type = str, default = 'cuda:4')
    parser.add_argument('--resume', type = int, default = 1)
    parser.add_argument('--roberta', type = int, default = 1)
    parser.add_argument('--model_with_sentiment', type = int, default = 0)
    parser.add_argument("--base_model",type = str, default = "hfl/chinese-bert-wwm-ext")
    parser.add_argument('--resume_checkpoint_path', type = str, default = "/home/liangwenjie/Agreement/runs/May12_14-14-37_gpu166round4_train_LR_1e-05_Batch_16_onlyAnswer_0_Label2_1_Resume_origin_choose")
    parser.add_argument('--save_checkpoint_path', type = str, default = "./None")
    parser.add_argument('--dropout', type = float, default = 0.2)

    args = parser.parse_args()
    # print(args.alpha)
    # Handle_OriginData(args)
    Train_Dataloader, Eval_Dataloader, label_count = Handle_OriginData(args)
    if args.generate == 1:
        test_generate(args, Train_Dataloader)
    else:
        train(args, Train_Dataloader, label_count, Eval_Dataloader)

def main_for_gen():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_only_answer', type = int, default = 0)
    parser.add_argument('--label2', type = int, default = 1)
    parser.add_argument('--data_path', type=str, default='./data/round4_test.csv')
    parser.add_argument('--generate', type = int, default = 1)
    # parser.add_argument('--data_vector_path', type=str, default='./weibo.json')
    parser.add_argument('--batch_size', type = int, default = 16)
    parser.add_argument('--num_workers', type = int, default = 0)
    parser.add_argument('--sentence_max_length', type = int, default = 60)
    # parser.add_argument('--sentiment_dict_path', type = str, default = '../data/Chinese_Corpus-master/sentiment_dict/ntusd/')
    # parser.add_argument('--using_Transformer', action = "store_true")
    parser.add_argument('--learning_rate', type = float, default = 1e-6)
    parser.add_argument('--epochs', type = int, default = 80)
    parser.add_argument('--device', type = str, default = 'cuda:0')
    parser.add_argument('--resume', type = int, default = 1)
    parser.add_argument('--roberta', type = int, default = 0)
    parser.add_argument('--model_with_sentiment', type = int, default = 0)
    parser.add_argument("--base_model",type = str, default = "bert-base-chinese")
    parser.add_argument('--resume_checkpoint_path', type = str, default = "/home/liangwenjie/Agreement/runs/May12_14-14-37_gpu166round4_train_LR_1e-05_Batch_16_onlyAnswer_0_Label2_1_Resume_origin_choose")
    parser.add_argument('--save_checkpoint_path', type = str, default = "./None")
    parser.add_argument('--dropout', type = float, default = 0.2)

    args = parser.parse_args()
    model_names = [
        'Eva',
        'Eva2',
        'CPM',
        'CDial-GPT',
        'Zhouwenwang'
    ]
    # with open("output_gen.txt",'a') as f:
  
    for model_name in model_names:
        for i in range(2):
            f = open("output_gen.txt",'a')
            f.write(model_name+" "+str(i)+"------------------------\n")
            f.close()
            print(model_name+" "+str(i)+"------------------------\n")
            path = os.path.join(".","genresult",model_name+"-prob_formats_1_gen","{}.csv".format(i))
            args.data_path = path
            Train_Dataloader, Eval_Dataloader, label_count = Handle_OriginData(args)
            if args.generate == 1:
                test_generate(args, Train_Dataloader)
            else:
                train(args, Train_Dataloader, label_count, Eval_Dataloader)
            f = open("output_gen.txt",'a')
            f.write("------------------------\n")
            f.close()
        
from transformers import pipeline

def my_sentiment(lines,pathh,args):
    js = []



    text_classification = pipeline('sentiment-analysis', model="techthiyanes/chinese_sentiment", return_all_scores = True, device = int(args.device.split(":")[1]) )
    for line_index in tqdm(range(0, len(lines))):
        line = lines[line_index]
        tmp = line.split("\n")[0].split("\t")
        # print(tmp)
        sentence = tmp[0].split("[SEP]")[1]
        # jss = tokenizer(sentence)
        # jss = model(torch.tensor([jss['input_ids']]).to(args.device),torch.tensor([jss['attention_mask']]).to(args.device),torch.tensor([jss['token_type_ids']]).to(args.device))
        jss = text_classification(sentence)[0]
        x = []
        for k in range(5):
            x.append(jss[k]['score'])

        js.append(x)
    json.dump(js,open(pathh,'w'))
    # assert 1 == 2


if __name__ == "__main__":
    # main()
    main()