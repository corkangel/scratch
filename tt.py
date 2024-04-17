import torch
from torch import nn

batch_size = 6
input_channels = 5
output_features = 2

input_data = ((torch.arange(0, 480) * 0.1).pow(1.5) * 0.1).reshape(batch_size,input_channels, 4, 4)

print (input_data)

conv_layer1 = nn.Conv2d(input_channels, output_features, kernel_size=3, stride=2, padding=1)
conv_layer1.weight.data.fill_(.01)
conv_layer1.bias.data.fill_(.0)
mid = conv_layer1(input_data)
print(f"mid shape: {mid.shape} mid: { mid}")

conv_layer2 = nn.Conv2d(2, 1, kernel_size=2, stride=2, padding=0)
conv_layer2.weight.data.fill_(.1)
conv_layer2.bias.data.fill_(.1)
output = conv_layer2(mid)
print(f"output shape: {output.shape} 0:{output[0][0][0][0]}")

mse = nn.MSELoss()

target = torch.ones(6, 1, 1, 1)
loss = mse(output, target);
loss.backward()

print(f"loss: {loss}")

print(f"Weights gradient shape: {conv_layer1.weight.grad.shape}")
print(f"Bias gradient shape: {conv_layer1.bias.grad.shape}")

