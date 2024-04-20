
# t_model_conv_layer_backwards1

import torch
import torch.nn as nn
import torch.optim as optim

# Create a single 1x8x8 image
input = (torch.arange(0., 25.)*0.1).reshape(1, 1, 5, 5)

# Create a Conv2d layer
conv = nn.Conv2d(1, 1, kernel_size=3, stride=2, padding=0)
conv.weight.data.fill_(0.1)
conv.bias.data.fill_(0.1)

# Pass the image through the Conv2d layer
output = conv(input)

# Register a hook on the output to print the gradients
output.register_hook(lambda grad: print("Gradients on the activations:", grad.shape, grad[0][0]))

# Create a target tensor
target = torch.tensor([1])

# Create a CrossEntropyLoss
criterion = nn.CrossEntropyLoss()

print ("Output shape", output.view(1, -1).shape)
print("Output ", output[0][0])

# Compute the loss
loss = criterion(output.view(1, -1), target)
print ("loss:", loss)

# Zero the gradients
conv.zero_grad()

# Backward pass
loss.backward()

# Print the gradients
for name, param in conv.named_parameters():
    print("Params Gradients", name, param.grad[0])

print ("Weights Before:", conv.weight.data[0])

# Create an optimizer
optimizer = optim.SGD(conv.parameters(), lr=0.01)

# Update the weights
optimizer.step()

print ("Weights After:", conv.weight.data[0])


