
# t_model_conv_layer_backwards
import torch
from torch import nn

batch_size = 1
input_channels = 1
output_features = 1

input_data = ((torch.arange(0, batch_size * 25) * 0.1).pow(1.5) * 0.1).reshape(batch_size, input_channels, 5, 5)

conv_layer1 = nn.Conv2d(input_channels, output_features, kernel_size=3, stride=2, padding=0)
conv_layer1.weight.data.fill_(.1)
conv_layer1.bias.data.fill_(.1)
mid = conv_layer1(input_data)
print(f"mid shape: {mid.shape} 0:{mid[0][0][0][0]} 1:{mid[0][0][0][1]} ")

print(f"conv_layer1.weight: {conv_layer1.weight}")
print(f"conv_layer1.weight.shape: {conv_layer1.weight.shape}")

softmax = nn.Softmax(dim=1)
smo = softmax(mid)
print ("mid", mid)
print ("smo", smo)

target = torch.tensor([1,0])
loss_fn = nn.CrossEntropyLoss()
loss = loss_fn(mid.squeeze(), target)
loss.backward()

print(f"loss: {loss}")
print ("conv_layer1.weight.grad", conv_layer1.weight.grad)
print(f"conv_layer1.weight: {conv_layer1.weight}")