import torch
from tqdm import tqdm
import torch.nn as nn
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch.optim.adam
import matplotlib.pyplot as plt
import time

class CSVDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# Define the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#Configs:

batch_size = 100
input_size = 2
num_classes = 2
num_epochs = 20
learning_rate = 0.001

activation_functions = {
    'ReLU': nn.ReLU(),
    # 'GELU': nn.GELU(),
    'Sigmoid': nn.Sigmoid()
}

optimizers = ['Adam', 'SGD']
learning_rates = [0.001, 0.1]

# Define the dataset class

preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

data = pd.read_csv('Dataset.csv')
data["target"].isnull().sum()

train_data , test_data = train_test_split(data , test_size = 0.2 , random_state=42)

train_data , valid_data = train_test_split(train_data , test_size = 0.25 , random_state=42)

x_train = torch.tensor(train_data.drop("target", axis=1).values, dtype=torch.float32)
y_train = torch.tensor(train_data["target"].values, dtype=torch.long)
x_valid = torch.tensor(valid_data.drop("target", axis=1).values, dtype=torch.float32)
y_valid = torch.tensor(valid_data["target"].values, dtype=torch.long)
x_test = torch.tensor(test_data.drop("target", axis=1).values, dtype=torch.float32)
y_test = torch.tensor(test_data["target"].values, dtype=torch.long)


train_data = CSVDataset(x_train, y_train)
valid_data = CSVDataset(x_valid, y_valid)   
test_data = CSVDataset(x_test, y_test)

# Define the DataLoader

train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=batch_size,
                                           shuffle=True)


validation_loader = torch.utils.data.DataLoader(dataset=valid_data, batch_size=batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size=batch_size,
                                          shuffle=False)

examples = iter(train_loader)
samples, labels = next(examples)

# Define the model

class MLP_Net(nn.Module):
  def __init__(self, num_classes,activation_fn):
    super(MLP_Net, self).__init__() 

    # self.l0 = nn.Linear()

    #Input Layer
    self.l1 = nn.Linear(x_train.shape[1], 100)
    self.bn1 = nn.BatchNorm1d(100)
    #Hidden Layer
    self.l2 = nn.Linear(100, 100)
    self.bn2 = nn.BatchNorm1d(100)
    self.l3 = nn.Linear(100, 50)
    self.bn3 = nn.BatchNorm1d(50)
    self.l4 = nn.Linear(50, 50)
    self.bn4 = nn.BatchNorm1d(50)
    #Output Layer
    self.l5 = nn.Linear(50, num_classes)


    self.activation = activation_fn
    self.sigmoid = nn.Sigmoid()
    # self.tanh = nn.Tanh()
    # self.gleu = nn.GELU()



  def forward(self, x):
    
    out = self.l1(x)    
    out = self.bn1(out)
    out = self.activation(out)

    
    out = self.l2(out)
    out = self.bn2(out)        
    out = self.activation(out)
    
    out = self.l3(out)
    out = self.bn3(out)  
    out = self.sigmoid(out)
    

    # out = self.l4(out)
    # out = self.bn4(out)    
    # out = self.activation(out)
    

    out = self.l5(out)

    return out
  


#Model:

model = MLP_Net(num_classes,activation_functions)
model = model.to(device)

#Train:

def train_and_evaluate(activation_fn, optimizer_name, lr):
    model = MLP_Net( num_classes, activation_fn).to(device)
    criterion = nn.CrossEntropyLoss()

    if optimizer_name == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    elif optimizer_name == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    elif optimizer_name == 'RMSprop':
        optimizer = torch.optim.RMSprop(model.parameters(), lr=lr)
    else:
        raise ValueError('Optimizer not supported')

    train_loss_history, train_accuracy_history = [], []
    val_loss_history, val_accuracy_history = [], []

    start_time = time.time()

    for epoch in range(num_epochs):
        model.train()
        train_loss, train_correct, train_total = 0, 0, 0

        for data, labels in train_loader:
            data = data.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0

        with torch.no_grad():
            for data, labels in validation_loader:
                data = data.to(device)
                labels = labels.to(device)
                outputs = model(data)
                val_loss += criterion(outputs, labels).item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        train_loss /= len(train_loader)
        train_accuracy = 100 * train_correct / train_total
        val_loss /= len(validation_loader)
        val_accuracy = 100 * val_correct / val_total

        train_loss_history.append(train_loss)
        train_accuracy_history.append(train_accuracy)
        val_loss_history.append(val_loss)
        val_accuracy_history.append(val_accuracy)

    elapsed_time = time.time() - start_time

    model.eval()
    n_correct, n_samples = 0, 0
    with torch.no_grad():
        for data, labels in test_loader:
            data = data.to(device)
            labels = labels.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs, 1)
            n_samples += labels.size(0)
            n_correct += (predicted == labels).sum().item()
    test_accuracy = 100.0 * n_correct / n_samples

    return {
        'train_loss': train_loss_history,
        'train_acc': train_accuracy_history,
        'val_loss': val_loss_history,
        'val_acc': val_accuracy_history,
        'test_acc': test_accuracy,
        'time': elapsed_time,
        'model': model
    }




results = {}

for act_name, act_fn in activation_functions.items():
    for opt_name in optimizers:
        for lr in learning_rates:
            print(f'Training with Activation={act_name}, Optimizer={opt_name}, LR={lr}')
            res = train_and_evaluate(act_fn, opt_name, lr)
            key = f'{act_name}_{opt_name}_lr{lr}'
            results[key] = res
            print(f"Test Accuracy: {res['test_acc']:.2f}%, Time: {res['time']:.2f}s\n")



plt.figure(figsize=(12,6))
names = list(results.keys())
test_accs = [results[k]['test_acc'] for k in names]

plt.barh(names, test_accs)
plt.xlabel('Test Accuracy (%)')
plt.title('Comparison of Test Accuracy for Different Configurations')
plt.grid(True)
plt.show()

#Ploting the decision boundary

def plot_decision_boundary(model, X, y, title="Decision Boundary"):
    model = model.to("cpu")
    model.eval()
    X_np = X.cpu().numpy()
    y_np = y.cpu().numpy()

    x_min, x_max = X_np[:, 0].min() - 1, X_np[:, 0].max() + 1
    y_min, y_max = X_np[:, 1].min() - 1, X_np[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                         np.linspace(y_min, y_max, 200))
    grid = np.c_[xx.ravel(), yy.ravel()]
    grid_tensor = torch.tensor(grid, dtype=torch.float32)

    with torch.no_grad():
        preds = model(grid_tensor)
        if preds.shape[1] == 1:
            preds = torch.sigmoid(preds).numpy().reshape(xx.shape)
        else:
            preds = torch.softmax(preds, dim=1)[:, 1].numpy().reshape(xx.shape)

    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, preds, levels=50, cmap="RdBu", alpha=0.6)
    plt.colorbar()
    plt.scatter(X_np[:, 0], X_np[:, 1], c=y_np, cmap="RdBu", edgecolor="k", s=40)
    plt.title(title)
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.show()


for key, result in results.items():
    model = result['model']
    plt.figure()
    plot_decision_boundary(model, x_test, y_test, title=f"Decision Boundary: {key}")


#compare diffrentent activation functions and optimizers

from sklearn.metrics import roc_curve, auc

plt.figure(figsize=(10, 8))
for key, result in results.items():
    model = result['model']
    model.eval()
    model = model.to("cpu")
    with torch.no_grad():
        outputs = model(x_test)
        if outputs.shape[1] == 1:
            probs = torch.sigmoid(outputs).cpu().numpy().flatten()
        else:
            probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy().flatten()
    fpr, tpr, _ = roc_curve(y_test.cpu().numpy(), probs)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'{key} (AUC = {roc_auc:.2f})')

plt.plot([0, 1], [0, 1], 'k--', label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for All Models')
plt.legend()
plt.grid(True)
plt.show()
