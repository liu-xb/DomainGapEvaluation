from MyDataset import MyDataset
import datetime, torch, os, torchvision
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import numpy as np
import torch.nn.functional as F
import time
import os,sys

def myprint(l, f):
	print l
	f.writelines(l + '\n')
	f.flush()

MY_GPU = [3]
EXTRA_WEIGHT = 64
MARGIN1 = 0.
MARGIN2 = 1
Auxiliary_BATCHSIZE = 32

EPOCH = 60
BATCH_SIZE = 32
LR = 0.00035
STEP_SIZE = 20
GAMMA = 0.1
TEST_BATCH_SIZE = 20

TRAIN_ROOT = '/home/xbliu/disk/cars196/car_ims/'#densepose/part/upper-new-id/'
TRAIN_TXT = '/home/xbliu/disk/cars196/train.txt'#densepose/part/train-file/'

TEST_ROOT = '/home/xbliu/disk/cars196/car_ims/'
TEST_TXT = '/home/xbliu/disk/cars196/test.txt'

WEIGHT = 'model/baseline-sgd-0.867.pt'
FINETUNE = 0
STRICT = 0

BEGIN_TIME = datetime.datetime.now()
today = str(BEGIN_TIME).split()[0]
if not os.path.exists('log/'+today):
	os.mkdir('log/' + today)
if not os.path.exists('snapshot/'+today):
	os.mkdir('snapshot/'+today)

SAVE_FILE = 'extra_-ln(1-p)_useVeRi_lossweight_' + str(EXTRA_WEIGHT)
SAVE_FILE += '_margin1_'  + str(MARGIN1) + '_margin2_' + str(MARGIN2)

LOG_TXT = 'log/' + today + '/' + SAVE_FILE + '.log'
SNAPSHOT = 'snapshot/' + today + '/' + SAVE_FILE + '-'

fid_log = open(LOG_TXT, 'w')
myprint(str(BEGIN_TIME), fid_log)
myprint('Log: ' + LOG_TXT, fid_log)
myprint('Snapshot: ' + SNAPSHOT, fid_log)
myprint('Extra_weight: ' + str(EXTRA_WEIGHT), fid_log)
myprint('Margin1:' + str(MARGIN1), fid_log)
myprint('Margin2:' + str(MARGIN2), fid_log)
myprint('auxiliarybatchsize:'+str(Auxiliary_BATCHSIZE), fid_log)
myprint('Finetune:'+str(FINETUNE), fid_log)

# device = torch.device('cuda: '+str(MY_GPU))
device = MY_GPU[0]

class MyLoss_v2(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, input):
        _, maxindex = torch.max(input.data, 1)
        maxindex = maxindex.unsqueeze(1)
        f = F.softmax(input, 1)
        f = f.gather(1, maxindex)
        f = -torch.log(-f+1)
        return f.mean()

class MyLoss(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, input):
        f = F.softmax(input, 1)
        f = f*torch.log(f)
        return f.mean(0).sum()

class MarketNet(nn.Module):
    def __init__(self):
        super(MarketNet, self).__init__()
        self.resnet_layer = torchvision.models.resnet50(pretrained=False)
        self.resnet_layer.load_state_dict(torch.load('/home/xbliu/market1501/pytorch/model/resnet50-19c8e357.pth'))
        self.fc = nn.Linear(self.resnet_layer.fc.in_features, 196)
        self.pool_bn = nn.BatchNorm1d(self.resnet_layer.fc.in_features)
        nn.init.constant_(self.pool_bn.weight, 1)
        nn.init.constant_(self.pool_bn.bias, 0)
        nn.init.normal_(self.fc.weight, std=0.001)
        nn.init.constant_(self.fc.bias, 0)
        self.resnet_layer = nn.Sequential(*list(self.resnet_layer.children())[:-2])

    def forward(self, x):
        x = self.resnet_layer(x)
        self.globalpooling = nn.AvgPool2d(kernel_size=(x.size()[2],x.size()[3]),stride = 1)
        x = self.globalpooling(x)
        x = x.view(x.size(0), -1)
        x = self.pool_bn(x)
        x = self.fc(x)
        return x

class DataProvider:
    def __init__(self, batch_size):
      self.batch_size = batch_size
      self.set = MyDataset(root = 'root-to-other-datasets', txt = 'image-list-for-other-datasets', transform=transform_train)
      self.iter = 0
      self.dataiter = None
    def build(self):
        dataloader = torch.utils.data.DataLoader(self.set, batch_size=self.batch_size,shuffle=True, num_workers=4)
        self.dataiter = torch.utils.data.dataloader._DataLoaderIter(dataloader)
    def next(self):
        if self.dataiter is None:
            self.build()
        try:
            batch = self.dataiter.next()
            self.iter += 1
            return batch
        except StopIteration:
            self.build()
            self.iter=1
            batch = self.dataiter.next()
            return batch

transform_train = transforms.Compose([
    transforms.Resize((224,224)),
    # transforms.RandomCrop((256,128), padding=10),
    # transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

transform_test = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

trainset = MyDataset(root = TRAIN_ROOT, txt = TRAIN_TXT, transform = transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

testset = MyDataset(root = TEST_ROOT, txt = TEST_TXT, transform = transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=TEST_BATCH_SIZE, shuffle=False, num_workers=4)

AuxiliaryLoader = DataProvider(Auxiliary_BATCHSIZE)

net = MarketNet()
if FINETUNE:
    net.load_state_dict(torch.load(WEIGHT), strict = STRICT)
net = net.to(device)
net = torch.nn.DataParallel(net, device_ids=MY_GPU, output_device=device)

criterion = nn.CrossEntropyLoss()
mycriterion = MyLoss()

param = []
params_dict = dict(net.named_parameters())
new_param = ['embed.weight', 'embed.bias', 'embed_bn.weight', 'embed_bn.bias', 'fc.weight', 'fc.bias']
for key, v in params_dict.items():
    if key in new_param:
        param += [{ 'params':v,  'lr_mult':1}]
    else:
        param += [{ 'params':v,  'lr_mult':1}]

# optimizer = optim.SGD(param, lr=LR, weight_decay=5e-4, momentum=0.9)
optimizer = optim.Adam(param, lr=LR, weight_decay = 5e-4)

current_num_iter = 0

myprint("Start Training Using GPU: " + str(MY_GPU), fid_log)

max_r1, max_map, max_acc, max_loss = 0., 0., 0., 0.
max_iter, max_auxiliary_eloss= 0., 0.

free_memory = os.popen('nvidia-smi --query-gpu=memory.free --format=csv,noheader').readlines()[device].split(' ')[0]
free_memory = float(free_memory)
NotOK = 0
if free_memory < 8000:
    NotOK = 1
while NotOK:
    print free_memory,'-',
    sys.stdout.flush()
    time.sleep(5)
    free_memory = os.popen('nvidia-smi --query-gpu=memory.free --format=csv,noheader').readlines()[device].split(' ')[0]
    free_memory = float(free_memory)
    if free_memory > 8000:
        NotOK = 0

for epoch in range( EPOCH):
    this_lr = LR * (GAMMA ** ((epoch+1)//STEP_SIZE))
    myprint('\nEpoch: ' + str(epoch+1) + ', Learning Rate: '+ str(this_lr), fid_log)
    for param_group in optimizer.param_groups:
		param_group['lr'] = param_group['lr_mult'] * this_lr

    net.train()
    sum_loss, sum_auxiliary_loss, correct, total = 0., 0., 0., 0.
    
    for i, data in enumerate(trainloader, 0):
        auxiliarydata, auxiliarylabel = AuxiliaryLoader.next()
        auxiliarydata = auxiliarydata.to(device)
        
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        
        outputs = net(torch.cat((inputs, auxiliarydata),0))#.to(device)

        loss = criterion(outputs[:inputs.shape[0]], labels)
        auxiliaryloss = mycriterion(outputs[inputs.shape[0]:])
        
        final_loss = loss + auxiliaryloss * EXTRA_WEIGHT         
        final_loss.backward()
        # loss.backward()
        optimizer.step()

        sum_loss += loss.item()
        sum_auxiliary_loss += auxiliaryloss.item()
        _, predicted = torch.max(outputs[:inputs.shape[0]].data, 1)
        total += labels.size(0)
        correct += predicted.eq(labels.data).cpu().sum()
        current_num_iter += 1
        if (current_num_iter%200 ==0) or (current_num_iter == 1):
            myprint('[epoch:' + str(epoch+1) + ', iter:' + str(current_num_iter)
            	+ '] Loss: ' + str(round(sum_loss/(i+1.0), 3)) 
            	+ ' | Acc: ' + str(round(100. * correct.item() / total,2)) 
            	+ ' | auxiliaryloss: ' + str(round(sum_auxiliary_loss/(1.0+i),3)), fid_log)

    if (epoch+1)%5 == 0:
        myprint('\nEpoch: ' + str(epoch+1) + ' Testing...', fid_log)
        with torch.no_grad():
            correct = 0
            total = 0
            sum_loss = 0
            for data in testloader:
                net.eval()
                images, labels = data
                images, labels = images.to(device), labels.to(device)
                outputs = net(images).to(device)
                loss = criterion(outputs, labels)
                sum_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum()
            myprint('Test accuracy: %.2f%%' % (100. * correct / total), fid_log)
            acc = 100. * correct.item() / total
            myprint('Test loss: %.3f' % (sum_loss / len(testloader)), fid_log)
            sum_loss /= len(testloader)
            myprint('Saving model: ' + SNAPSHOT +str(epoch+1).zfill(3)+'.pt', fid_log)
            torch.save(net.to('cpu').module.state_dict(), SNAPSHOT + str(epoch+1).zfill(3) + '.pt')
            myprint('Cost' + str(datetime.datetime.now() - BEGIN_TIME), fid_log)
            net = net.to(device)

myprint("Training Finished, Total EPOCH=" + str(EPOCH), fid_log)
myprint('Cost ' + str(datetime.datetime.now() - BEGIN_TIME), fid_log)
myprint(str(datetime.datetime.now()), fid_log)
fid_log.close()
