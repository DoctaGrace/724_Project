import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

# Class containing logic for multihead self attention
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, in_channels, out_channels, num_heads):
        super(MultiHeadSelfAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = out_channels // num_heads

        # Query, key, value matrices
        self.W_q = nn.Linear(in_channels, out_channels, bias=False)
        self.W_k = nn.Linear(in_channels, out_channels, bias=False)
        self.W_v = nn.Linear(in_channels, out_channels, bias=False)
        self.fc = nn.Linear(out_channels, out_channels, bias=False)

    def forward(self, x):
        N, C, H, W = x.size()
        x = x.view(N, C, H * W).permute(0, 2, 1)  

        queries = self.W_q(x)
        keys = self.W_k(x)
        values = self.W_v(x)

        queries = queries.view(N, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        keys = keys.view(N, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        values = values.view(N, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        energy = torch.matmul(queries, keys.permute(0, 1, 3, 2))
        attention = torch.softmax(energy, dim=-1)
        out = torch.matmul(attention, values).permute(0, 2, 1, 3).contiguous()
        out = out.view(N, -1, self.head_dim * self.num_heads)
        out = self.fc(out)
        out = out.view(N, 128, H, W)  
        return out


# Model definitiion
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 128, 4, 1) 
        self.self_attention1 = MultiHeadSelfAttention(128, 128, 4) 
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(18432, 256)  
        self.fc2 = nn.Linear(256, 256)

    # Forward pass
    def forward(self, x):
        x = self.conv1(x) 
        x = nn.functional.relu(x)
        x = nn.functional.max_pool2d(x, 2)
        x = self.self_attention1(x) 
        x = nn.functional.relu(x)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = nn.functional.relu(x)
        x = self.dropout1(x)
        x = self.fc2(x)

        output = nn.functional.log_softmax(x, dim=1)
        return output


    
# Load MNIST dataset
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=1000, shuffle=False)

# Initialize the CNN
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = Net().to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)

# Train the network
num_epochs = 10

for epoch in range(num_epochs):
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f'Epoch {epoch + 1}, Loss: {running_loss / len(trainloader)}')

print('Training finished.')

# Test the network
correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        images, labels = images.to(device), labels.to(device)

        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy on the test set: {100 * correct / total}%')
