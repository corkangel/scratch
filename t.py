import torch
import torch.nn as nn

# Define the batch size, input channels, output features, stride and padding
batch_size = 6
input_channels = 5
output_features = 4
stride = 2
padding = 1

input_data = torch.ones(6, input_channels, 8, 8)
conv_layer = nn.Conv2d(input_channels, 4, kernel_size=3, stride=2, padding=1)
conv_layer.weight.data.fill_(.1)
conv_layer.bias.data.fill_(.1)

# Pass the input data through the convolutional layer
output_data = conv_layer(input_data)

print(f"Output shape: {output_data.shape}")
print(f"Output data: {output_data}")
