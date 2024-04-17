import torch
from torch import nn

batch_size = 2
input_channels = 2
output_features = 4
stride = 2
padding = 1

input_data = torch.ones(batch_size, input_channels, 4, 4)
conv_layer = nn.Conv2d(input_channels, output_features, kernel_size=3, stride=2, padding=1)
conv_layer.weight.data.fill_(.1)
conv_layer.bias.data.fill_(.1)

# Pass the input data through the convolutional layer
output_data = conv_layer(input_data)

print(f"Output shape: {output_data.shape}")
print(f"Output data: {output_data}")