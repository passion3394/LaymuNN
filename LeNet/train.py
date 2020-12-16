import torch
import torchvision
import torch.nn as nn
from model import LeNet
import torch.optim as optim
import torchvision.transforms as transforms
import os

#key step 1
torch.distributed.init_process_group(backend="nccl")

#key step 2
local_rank = torch.distributed.get_rank()
torch.cuda.set_device(local_rank)
device = torch.device('cuda', local_rank)

transform = transforms.Compose(
    [transforms.ToTensor(),transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))]
    )

train_set = torchvision.datasets.CIFAR10(root='./data',train=True,download=True,transform=transform)
train_loader = torch.utils.data.DataLoader(train_set,batch_size=200,num_workers=4,sampler=torch.utils.data.distributed.DistributedSampler(train_set))
val_set = torchvision.datasets.CIFAR10(root='./data',train=False,download=True,transform=transform)
val_loader = torch.utils.data.DataLoader(val_set,batch_size=10000,num_workers=4,sampler=torch.utils.data.distributed.DistributedSampler(val_set))
val_data_iter = iter(val_loader)
val_image,val_label = val_data_iter.next()
classes = ('plane', 'car', 'bird', 'cat', 'deer',\
           'dog', 'frog', 'horse', 'ship','truck')

net = LeNet()
net.to(device)
if torch.cuda.device_count() > 1:
  print('We use ', torch.cuda.device_count(), ' GPUs')
  net = torch.nn.parallel.DistributedDataParallel(net, device_ids=[local_rank], output_device=local_rank)
loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(),lr=0.001)

for epoch in range(50):
  run_loss = 0.0
  for step,data in enumerate(train_loader, start=0):
    inputs, labels = data
    if torch.cuda.is_available():
      inputs = inputs.to(device)
      labels = labels.to(device)
    
    #the gradient to cleared
    optimizer.zero_grad()
    #forward + backward + optimize
    outputs = net(inputs)
    loss = loss_function(outputs,labels)
    loss.backward()

    optimizer.step()
    
    run_loss += loss.item()
    print('[%d, %5d] train_loss: %.3f' %(epoch + 1, step + 1, run_loss))

  if local_rank ==0:
    with torch.no_grad():
      if torch.cuda.is_available():
        val_image = val_image.to(device)
        val_label = val_label.to(device)
      outputs = net(val_image)
      predict_y = torch.max(outputs, dim=1)[1]
      accuracy = (predict_y == val_label).sum().item() / val_label.size(0)

      print('[%d, %5d] train_loss: %.3f test_accuracy: %.3f' %(epoch + 1, step + 1, run_loss / 100, accuracy))
      run_loss = 0.0

print('Training Done.')

if local_rank == 0:
  #save model
  save_path = 'ckpt/'
  if not os.path.exists(save_path):
    os.makedirs(save_path)
  save_ckpt_path = os.path.join(save_path, 'LeNet.pth')
  print('saved path: ',save_ckpt_path)
  torch.save(net.state_dict(), save_ckpt_path)
