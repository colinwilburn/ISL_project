import torch
from torch import nn
import time
import plotly.graph_objects as go
from loader import get_train_val_loaders, get_test_loader

class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 4),
            nn.ReLU(),
        )

    def forward(self, features):
        logits = self.linear_relu_stack(features)
        return logits


def train_loop(train_loader, model, loss_fn, optimizer):
    size = len(train_loader.dataset)
    correct = 0
    for batch, (features, target) in enumerate(train_loader):
        features = features.to(device)
        target = target.to(device)       
        pred = model(features)
        loss = loss_fn(pred, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        correct += (pred.argmax(1) == target).type(torch.float).sum().item()

    accuracy = correct / size
    loss = loss.item()
    return loss, accuracy

def test_loop(loader, model):
    size = len(loader.dataset)
    correct = 0
    with torch.no_grad():
        for features, target in loader:
            features = features.to(device)
            target = target.to(device) 
            pred = model(features)
            correct += (pred.argmax(1) == target).type(torch.float).sum().item()
    accuracy = correct / size
    return accuracy

if torch.cuda.is_available():
    device = 'cuda'

model = NeuralNet()
'''
if torch.cuda.device_count()>1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    model = nn.DataParallel(model, device_ids=range(torch.cuda.device_count))
'''
model.to(device)


learning_rate = 1e-3
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

train_loader, val_loader = get_train_val_loaders()
test_loader = get_test_loader()




epochs = int(1e3)

acc_train_list = []
acc_val_list = []
acc_test_list = []

start = time.time()
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    loss_train, accuracy_train = train_loop(
        train_loader, model, loss_fn, optimizer
    )
    print("Training:")
    print(f"\tloss: {loss_train:>7f}")
    print(f"\taccuracy: {accuracy_train:>4f}")
    acc_train_list.append(accuracy_train)

    accuracy_val = test_loop(val_loader, model)
    print("Validation:")
    print(f"\taccuracy: {accuracy_val:>4f}")
    acc_val_list.append(accuracy_val)

    accuracy_test = test_loop(test_loader, model)
    print("Test:")
    print(f"\taccuracy: {accuracy_test:>4f}")
    acc_test_list.append(accuracy_test)

print(f"Time to run: {time.time() - start}")

x = list(range(1, epochs+1))

fig = go.Figure()
fig.add_trace(go.Scatter(
    x = x,
    y = acc_train_list,
    mode='markers+lines',
    name = "Training Accuracy"
))
fig.add_trace(go.Scatter(
    x = x,
    y = acc_val_list,
    mode='markers+lines',
    name='Validation Accuracy'
))
fig.add_trace(go.Scatter(
    x = x,
    y = acc_test_list,
    mode='markers+lines',
    name='Test Accuracy'
))
fig.update_layout(
    xaxis_title='Epoch',
    yaxis_title='Accuracy',
    title="Neural Network Accuracy during Training",
    title_x=0.5,
    title_y=0.97,
    font=dict(family="Courier New, monospace",size=14),
    legend=dict(
        yanchor="top",
        y=0.99,
        xanchor="left",
        x=0.01
    )
)
#fig.update_xaxes(type="log")
fig.show(renderer='firefox')