
# t_model_conv_layer_backwards2

import torch
import torch.nn as nn
import torch.optim as optim

# Create a single 1x8x8 image
input = (torch.arange(0., 81.)*0.1).reshape(1, 1, 9, 9)

# Create a Conv2d layer
conv1 = nn.Conv2d(1, 1, kernel_size=3, stride=2, padding=0)
conv1.weight.data.fill_(0.1)
conv1.bias.data.fill_(0.1)

conv2 = nn.Conv2d(1, 1, kernel_size=3, stride=2, padding=1)
conv2.weight.data.fill_(0.1)
conv2.bias.data.fill_(0.1)

# Pass the image through the Conv2d layer
mid = conv1(input)
output = conv2(mid)

# Register a hook on the output to print the gradients
mid.register_hook(lambda grad: print("Gradients on the mid activations:", grad.shape, grad[0][0]))
output.register_hook(lambda grad: print("Gradients on the output activations:", grad.shape, grad[0][0]))

# Create a target tensor
target = torch.tensor([1])

# Create a CrossEntropyLoss
criterion = nn.CrossEntropyLoss()

print ("Mid shape", mid.view(1, -1).shape)
print("Mid ", mid[0][0])

print ("Output shape", output.view(1, -1).shape)
print("Output ", output[0][0])

# Compute the loss
loss = criterion(output.view(1, -1), target)
print ("loss:", loss)

# Zero the gradients
conv1.zero_grad()
conv2.zero_grad()

# Backward pass
loss.backward()

# Print the gradients

for name, param in conv1.named_parameters():
    print("Params1 Gradients", name, param.grad[0])

for name, param in conv2.named_parameters():
    print("Params2 Gradients", name, param.grad[0])

print ("Weights Before:", conv1.weight.data[0])

# Create an optimizer
optimizer = optim.SGD(list(conv1.parameters()) + list(conv2.parameters()), lr=0.01)

# Update the weights
optimizer.step()

print ("Weights After:", conv1.weight.data[0])


