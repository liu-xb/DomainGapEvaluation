import sys
sys.path.append('/home/xbliu/')
from MyDataset import MyDataset
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np
import scipy.io as scio
import os
from sklearn import preprocessing
import multiprocessing
import torch.nn.functional as F
from sklearn import preprocessing
import multiprocessing
import time

MY_GPU = 3
SNAPSHOT = 'xxx.pt'
ID_NUM = 196
WAIT_LEVEL = 4
NEED_MEMORY = 2500
num_iter = 99
num_image = 50

print 'Using GPU:', str(MY_GPU), ' Snapshot:', SNAPSHOT

fid = open('/home/xbliu/dataset_path.txt','r')
dataset_path = fid.read()
dataset_path = eval(dataset_path)
fid.close()

device = MY_GPU
free_memory = os.popen('nvidia-smi --query-gpu=memory.free --format=csv,noheader').readlines()[device].split(' ')[0]
free_memory = float(free_memory)
NotOK = 0
if free_memory < NEED_MEMORY:
    NotOK = WAIT_LEVEL
    print 'waiting for gpu',device
while NotOK:
    print WAIT_LEVEL,
    sys.stdout.flush()
    time.sleep(5)
    free_memory = os.popen('nvidia-smi --query-gpu=memory.free --format=csv,noheader').readlines()[device].split(' ')[0]
    free_memory = float(free_memory)
    if free_memory > NEED_MEMORY:
        NotOK -= 1

#######################################################################
######################## Model defination #############################
#######################################################################
class MarketNet(nn.Module):
    def __init__(self):
        super(MarketNet, self).__init__()
        self.resnet_layer = torchvision.models.resnet50(pretrained=False)
        self.fc = nn.Linear(self.resnet_layer.fc.in_features, ID_NUM)
        
        self.pool_bn = nn.BatchNorm1d(self.resnet_layer.fc.in_features)
        self.resnet_layer = nn.Sequential(*list(self.resnet_layer.children())[:-2])
    
    def forward(self, x):
        x = self.resnet_layer(x)
        self.globalpooling = nn.AvgPool2d(kernel_size=(x.size()[2],x.size()[3]),stride = 1)
        x = self.globalpooling(x)
        x = x.view(x.size(0), -1)
        x = self.pool_bn(x)
        x2 = self.fc(x)
        return x,x2
#######################################################################
######################## Some Settings# ###############################
#######################################################################
device = torch.device('cuda: '+str(MY_GPU))
if not os.path.exists('cls'):
	os.mkdir('cls')
if not os.path.exists('feature'):
	os.mkdir('feature')
transform_test = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

net = MarketNet()
net.load_state_dict(torch.load(SNAPSHOT), strict=1)
net = net.to(device)

#######################################################################
#################### Start Extracting Features ########################
#######################################################################

test_set = ['aircraft', 'cars196', 'cats', 'cityscapes', 'coco', 'compcars', 'cub', 'cuhk03',
    'deepfashion', 'duke', 'fashionai', 'food', 'fruit', 'imagenet_test', 'lfw', 'market', 'mars',
    'monkey', 'msmt', 'oxford_flower_102', 'place365', 'pubfig83', 'stanford_dog', 'vehicleid',
    'voc12_person', 'yahoo_shoes' ]

with torch.no_grad():
    for i in range(len(test_set)):
        this_dataset = test_set[i]
        print this_dataset
        if this_dataset+'_test' in dataset_path:
        	this_root = dataset_path[this_dataset+'_root']
        	this_text = dataset_path[this_dataset+'_test']
        elif this_dataset + '_gallery' in dataset_path:
        	this_root = dataset_path[this_dataset+'_root']
        	this_text = dataset_path[this_dataset+'_gallery']
        else:
        	print 'no',this_dataset
        	continue
        if os.path.exists('feature/'+this_dataset+'_feature.mat'):
        	if os.path.exists('cls/'+this_dataset+'_feature.mat'):
        		continue
        queryfeature = []
        querycls = []
        queryset = MyDataset(root = this_root, txt = this_text, transform=transform_test)
        queryloader = torch.utils.data.DataLoader(queryset, batch_size=num_image, shuffle=False, num_workers=16)
        i=0

        for data in queryloader:
            i+=1
            net.eval()
            images, labels = data
            images = images.to(device)
            outputs1 = net(images)
            queryfeature.extend(list(outputs1[0].to('cpu').numpy()))
            querycls.extend(list(outputs1[1].to('cpu').numpy()))
            # print i
            if i>num_iter:
            	break
        queryfeature = np.array(queryfeature)
        queryfeature = np.transpose(queryfeature)

        querycls = np.array(querycls)
        querycls = np.transpose(querycls)
        scio.savemat('feature/'+this_dataset+'_feature.mat', {this_dataset+'_feature':queryfeature})
        scio.savemat('cls/'+this_dataset+'_feature.mat', {this_dataset+'_feature':querycls})

    
os.system('cp ~/disk/out_distri_detect/test_cls* cls/')
os.system('cp ~/disk/out_distri_detect/test_feat* feature/')
