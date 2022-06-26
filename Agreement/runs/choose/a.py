

from tensorboard.backend.event_processing import event_accumulator  # 导入tensorboard的事件解析器
# import xlsxwriter
import numpy as np

def Read_Tensorboard(path):  # path为tensoboard文件的路径
    ea = event_accumulator.EventAccumulator(path)  # 初始化EventAccumulator对象
    ea.Reload()  # 将事件的内容都导进去
    print(ea.scalars.Keys())
    loss_train = ea.scalars.Items("Loss/train")  # 根据上面打印的结果填写
    # val_IoU = ea.scalars.Items("val/IoU.leakage")
    # val_mAcc = ea.scalars.Items("val/mAcc")
    # learning_rate = ea.scalars.Items("learning_rate")
    # print(val_dice)
    # print(len(val_dice))

    # print([(i.step,i.value) for i in val_dice])
    train = []
    for i in range(len(loss_train)):
        train.append(loss_train[i].value)
    # Epoch=[]
    # Dice=[]
    # IoU=[]
    # Acc=[]
    # RL=[]
    # for i in range(200):
    #     print(i+1,val_dice[i].value,val_IoU[i].value,val_mAcc[i].value,learning_rate[i].value)
    #     Epoch.append(i+1)
    #     Dice.append(val_dice[i].value)
    #     IoU.append(val_IoU[i].value)
    #     Acc.append(val_mAcc[i].value)
    #     RL.append(learning_rate[i].value)
    # return Epoch,Dice,IoU,Acc,RL
    return train

# 把列表Epoch, Dice, IoU, Acc, RL 都写入名为filename的excecl文件当中
def write_PR(train,filename):
    #参数p/r/t是等长的数组，p表示presion,r是recall，t是阈值
    # # workbook = xlsxwriter.Workbook(filename)
    # worksheet = workbook.add_worksheet()

    # worksheet.activate()  # 激活表
    # title = ['train_loss'] # 设置表头
    # worksheet.write_row('A1',title) # 从A1单元格开始写入表头


    #  # Start from the first cell below the headers.
    # n_row = 2 #从第二行开始写
    import numpy as np
    f = open(filename,'w')
    for i in range(0,len(train),80):
        insertData=np.average(np.array(train[i:i+80]))
        f.write(str(insertData)+"\n")
    #     row = 'A' + str(n_row)
    #     worksheet.write_row(row, insertData)
    #     n_row=n_row+1
    # workbook.close()

if __name__ == '__main__':
    path = r"/home/liangwenjie/Agreement/runs/choose/chinese-roberta-wwm-ext/events.out.tfevents.1653662119.gpu166.9168.0"
    train=Read_Tensorboard(path)
    # print(Epoch, Dice, IoU, Acc, RL)
    write_PR(train,'adagrad.txt')
