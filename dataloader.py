import torch
import numpy as np
import glob
import scipy
import scipy.io as sio
from ModelResNet import *
from Preper import *
from torch.utils.data import Dataset, DataLoader
Y=Convolution_opMS(imgNonOlive,(16,16),(12,12))#appel a la fonction de la preparation de donnèes
Y=Y.tolist() #list of samples  'non olive'
A=Convolution_opMS(imgolive,(16,16),(5,5))#appel a la fonction de la preparation de donnèes
A=A.tolist()  #list of samples  'olive'
#len(X)
#savemat("olivecoup.mat",{"fo":X})
#print('number of data :' ,len(X)+len(Y))
z=np.concatenate((Y,A))
z=z.tolist()  #list of samples  'alldata
labels=[]
alldata = z
net.eval()
class ImageDataset:
    def __init__(self,alldata,  transform=None):    
        #self.root_dir = root_dir
        #load data from mat files
        self.alldata=alldata
        #glob.glob(self.root_dir+"/*.mat")
        alldata_olivier = A
        alldata_non_olivier = Y
        #transform data to unique list of data
        
        #a list of label 
        #à) construire à partir [0 ou 1] selon self.alldata
        
        for i in range(5027) : #number of samples 'non olive'
            labels.append(0)
        for i in range(4990) : #number of samples 'olive
            labels.append(1)
        
        #shuffle data using sklearn
        
        
        self.numdata = 10017
        self.transform = transform
        
    def __len__(self):
        return self.numdata
    def __getitem__(self, idx):
        label=labels[idx]

        #newidx = self.shuffle[idx]
        image = self.alldata[idx]
        label=np.asarray(label)
        #transform data from numpy to torch tensor
        imageTensor =np.asarray(alldata)# 
        imageTensor =torch.from_numpy(imageTensor)
        #plt.imshow(imageTensor[:,:,0])
        labelTensor =np.asarray(labels)# torch.from_numpy(label)
        labelTensor =torch.from_numpy(labelTensor)
        #print(imageTensor) 
        return imageTensor , labelTensor
if __name__ == '__main__':
    k= ImageDataset(z)
    k.__getitem__(60)
    
import sklearn.model_selection as model_selection
X_train, X_test, y_train, y_test = model_selection.train_test_split(alldata, labels, train_size=0.8,test_size=0.2)

class trainData():
    
    def __init__(self, X_data, y_data):
        self.X_data = X_data
        self.y_data = y_data
        
    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]
        
    def __len__ (self):
        return len(self.X_data)


train_data = trainData(torch.FloatTensor(X_train), 
                       torch.FloatTensor(y_train),
                       )

class testData():

    def __init__(self, X_data):
        self.X_data = X_data
        
    def __getitem__(self, index):
        return self.X_data[index]
      
    def __len__ (self):
        return len(self.X_data)
train_loader = DataLoader(dataset=train_data, batch_size=8, shuffle=True)
net.to(device)
device=torch.device("cuda:0" if torch.cuda.is_available () else "cpu")
test_data = testData(torch.FloatTensor(X_test))
EPOCHS =116
LEARNING_RATE = 0.001
def binary_acc(y_pred, y_testt):
    y_pred_tag = torch.round(y_pred)
    correct_results_sum = (y_pred_tag[:,0]  == y_testt).sum().float()
    acc = correct_results_sum/y_testt.shape[0]
    acc = torch.round(acc * 100)
    return acc


net.train()

for e in range(1, EPOCHS+1):
    epoch_loss = 0
    epoch_acc = 0
    for X_batch, y_batch in train_loader:
        
        
        X_batch =np.asarray(X_batch)
        X_batch =torch.from_numpy(X_batch)
        optimizer.zero_grad()
        
        X_batch= X_batch.permute(0,3,2,1).float()
        y_batch =np.asarray(y_batch) 
        y_batch =torch.from_numpy(y_batch)
        X_batch, y_batch =X_batch.to(device) , y_batch.to(device)
        y_pred = net(X_batch) 
        
        acc = binary_acc(y_pred, y_batch)
        
        loss = loss_fn(y_pred[:,0], y_batch.float())  
            
        loss.backward()
        
        optimizer.step()
        
        epoch_loss += loss.item()
        epoch_acc += acc.item()
        
    print('debut training : ')   
    print(f'Epoch {e+0:03}: | Loss: {epoch_loss/len(train_loader):.5f} | Acc: {epoch_acc/len(train_loader):.3f}')

"""
        if e % 70 == 69:
            torch.save(net.state_dict(), 'train_valid_exp4-epoch{}.pth'.format(e))
"""
#------------------------TEST---------------
test_loader = DataLoader(dataset=test_data)
y_pred_list = []
net.eval()
with torch.no_grad(): 
    for X_batch1 in test_loader:      
        X_batch1= X_batch1.permute(0,3,2,1).float()
        X_batch1=X_batch1.to(device) 
        y_test_pred =net(X_batch1)
        y_pred_tag = torch.round(y_test_pred[0,:])
        y_pred_list.append(y_pred_tag)
y_pred_list = [a.squeeze().tolist() for a in y_pred_list]
y_pred_list =np.asarray(y_pred_list)

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred_list[:,0].tolist())
error=(cm[0,1]+cm[1,0])/2004
acc=(cm[0,0]+cm[1,1])/2004
print(acc)

N=opMS(test,(16,16),(16,16))
lab=[]
res=[]

from ModelResNet import *
from Preper import *
#test
for i in range(len(N)):
    J=np.asarray(N[i])
    J=J.astype(np.float32)
    IMG1 =torch.from_numpy(J)
    IMG1=IMG1.unsqueeze_(0)
    IMG1= IMG1.permute(0,3,2,1)
    IMG1=IMG1.to(device)
    lab.append(torch.round(net(IMG1)).tolist())
    
    #print ('N[',i,']',' ; ', lab[i])
    if (lab[i]==[[0.0,1.0]]):
        res.append(0)
    elif (lab[i]==[[1.0,0.0]]):
        res.append(1)
    J=[]

n=np.asarray(res)
#10980/16=687
imgfinal=np.reshape(res, (686,686))
matplotlib.pyplot.imshow(imgfinal)


plt.savefig('res2.png')


