import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.utils as utils
from torchvision.datasets import ImageFolder
import torch.optim as optim

pixel_size = (32 , 32) # Standard Image size for the CIFAR model

transformer = transforms.Compose([
    transforms.Resize(pixel_size) ,
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)),
])

train_set= ImageFolder("cifar10/train" , transform=transformer)
val_set= ImageFolder("cifar10/val" , transform=transformer)
test_set = ImageFolder("cifar10/test" , transform=transformer)

train_data = utils.data.DataLoader(train_set , batch_size=32 , shuffle=True)

class CifarModel(nn.Module):
    def __init__(self):
        super().__init__()
        #(input_size-kernel)/stride+1,out=(out_channels,out_size,out_size)
        self.conv1 = nn.Conv2d(3 , 12 , 5)# (32 - 5)/1 + 1 = (24 , 28 , 28)
        self.pool = nn.MaxPool2d(2 , 2) # (24 , 14 , 14)
        self.conv2 = nn.Conv2d(12 , 24 , 5 ) # (12 , 10 , 10) -> Pool (12 , 5 , 5) -> Flatten -> (24 * 5 * 5)
        self.fc1 = nn.Linear(24*5*5 , 120)
        self.fc2 = nn.Linear(120 , 84)
        self.fc3 = nn.Linear(84 , 10)
        self.relu = nn.ReLU()

    def forward(self , X):
        out = self.relu(self.conv1(X)) 
        out = self.pool(out) 
        out = self.relu(self.conv2(out))
        out = self.pool(out) 
        out = torch.flatten(out , 1)
        out = self.fc1(out) 
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        
        return out#torch.softmax(out , 0)

model = CifarModel()

optimizer = optim.SGD(params=model.parameters() , lr=0.001 , momentum=0.9)

loss_fn = nn.CrossEntropyLoss()

epochs = 50
class_names = ["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]

for epoch in range(epochs):
    print(f"Epoch {epoch}/{epochs} ...")
    for i , data in enumerate(train_data):
        inputs , labels = data

        y_pred = model(inputs)

        loss = loss_fn(y_pred , labels)

        #print("Loss : " , loss)
        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

torch.save(model.state_dict() , "cifar10_trained.pth")

trained_weights = torch.load("cifar10_trained.pth")
# Load the weights in the model
model.load_state_dict(trained_weights)

model.eval()

test_data = utils.data.DataLoader(test_set , batch_size=32 , shuffle=True )

total = 0
correct = 0

with torch.no_grad():
    for data in test_data:
        image , label = data
        
        outputs = model(image)

        _ , predicted = torch.max(outputs , 1)
        
        total += label.size()[0]
        
        correct += (predicted == label).sum().item()
        
accuracy = 100 * correct / total

print(f"Accuracy : {accuracy}%" )





