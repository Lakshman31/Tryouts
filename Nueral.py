import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn

plt.title("Multi-class data, 4 informative features, 3 classes", fontsize="large")
X, Y = make_classification(
    n_features=4, n_redundant=0, n_informative=3, n_clusters_per_class=1, n_classes=3
)
plt.scatter(X[:, 0], X[:, 1], marker="o", c=Y, s=25, edgecolor="k")


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33, random_state=42)
X_train.shape, X_test.shape, Y_train.shape, Y_test.shape

X_train = X.reshape(-1, X.shape[1]).astype('float32')
Y_train = Y

X_test = X_test.reshape(-1, X_test.shape[1]).astype('float32')
Y_test = Y_test


Y_test = torch.from_numpy(X_test)
Y_test = torch.from_numpy(np.asarray(Y_test))


class Data(Dataset):
    def __init__(self):
        self.X=torch.from_numpy(X_train)
        self.Y=torch.from_numpy(Y_train)
        self.len=self.X.shape[0]
    def __getitem__(self,index):
        return self.X[index], self.Y[index]
    def __len__(self):
        return self.len

data=Data()
loader=DataLoader(dataset=data,batch_size=64)
print(data.X[0:5])
print(data.X.shape)
print(data.Y[0:5])
print(data.Y.shape)

input_dim=4     # how many Variables are in the dataset
hidden_dim = 25 # hidden layers
output_dim=3    # number of classes

class Net(nn.Module):
    def __init__(self, input, H, output):
        super(Net, self).__init__()
        self.linear1 = nn.Linear(input, H)
        self.linear2 = nn.Linear(H, output)

    def forward(self, x):
        x = torch.sigmoid(self.linear1(x))
        x = self.linear2(x)
        return x

clf = Net(input_dim, hidden_dim, output_dim)
print(clf.parameters)
print('W:', list(clf.parameters())[0].size())
print('b', list(clf.parameters())[1].size())

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(clf.parameters(), lr=0.1)

n_epochs = 1000
loss_list = []

for epoch in range(n_epochs):
    for x, y in loader:
        # clear gradient
        optimizer.zero_grad()
        # make a prediction
        z = clf(x)
        y=y.type(torch.LongTensor)
        loss = criterion(z,y)
        # calculate gradients of parameters
        loss.backward()
        # update parameters
        optimizer.step()

        loss_list.append(loss.data)

learning_rate = 1e-1
loss_list = []
for t in range(1000):
    y_pred = clf(x)
    loss = criterion(y_pred, y)
    loss_list.append(loss.item())
    clf.zero_grad()
    loss.backward()
    with torch.no_grad():
        for param in clf.parameters():
            param -= learning_rate * param.grad

step = np.linspace(0,1000,1000)
plt.title("step-wise loss")
plt.plot(step,np.array(loss_list))

params = list(clf.parameters())
w = params[0].detach().numpy()[0]
b = params[1].detach().numpy()[0]
t= params[3].detach().numpy()[0]
plt.scatter(X[:, 0], X[:, 1], c=Y,cmap='jet')

u = np.linspace(X[:, 0].min(), X[:, 0].max(), 2)
plt.title("decision boundaries")
plt.plot(u, (0.5-b-w[0]*u)/w[1])
plt.plot(u, (0.5-t-w[0]*u)/w[1])
plt.xlim(X[:, 0].min()-0.5, X[:, 0].max()+0.5)
plt.ylim(X[:, 1].min()-0.5, X[:, 1].max()+0.5)
plt.show()

x_val = torch.from_numpy(X_test)
z=clf(x_val)
yhat=torch.max(z.data,1)
print(yhat[1])
