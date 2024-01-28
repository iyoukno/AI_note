"""
yolo
train: test: val 7:2:1 不重复划分
"""
import os
import random
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--xml_path', default=r'D:\datasets\insulator\Annotations', type=str, help='input xml label path')
parser.add_argument('--txt_path', default=r'D:\datasets\insulator', type=str, help='output txt label path')
opt = parser.parse_args()

train_percent = 0.7
test_percent = 0.2  #这里的train_percent 是指占trainval_percent中的
xmlfilepath = opt.xml_path
txtsavepath = opt.txt_path
total_xml = os.listdir(xmlfilepath)
if not os.path.exists(txtsavepath):
    os.makedirs(txtsavepath)

num = len(total_xml)
list_index = range(num)

random.shuffle(total_xml)
train, test, val = range(int(num * train_percent)), range(int(num * train_percent), int(num * (train_percent + test_percent))+1),\
    range(int(num * (train_percent + test_percent))+1, num)

# file_trainval = open(txtsavepath + '/trainval.txt', 'w')
file_test = open(txtsavepath + '/test.txt', 'w')
file_train = open(txtsavepath + '/train.txt', 'w')
file_val = open(txtsavepath + '/val.txt', 'w')

for i in list_index:
    name = total_xml[i][:-4] + '\n'
    if i in train:
        file_train.write(name)
    elif i in test:
        file_test.write(name)
    else:
        file_val.write(name)


# file_trainval.close()
file_train.close()
file_val.close()
file_test.close()
