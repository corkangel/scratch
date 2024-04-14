import torch
import torch.nn as nn

# Define the batch size, input channels, output features, stride and padding
batch_size = 64
input_channels = 3
output_features = 4
stride = 2
padding = 1

input_data = torch.randn(64, 3, 28, 28)
conv_layer = nn.Conv2d(3, 4, kernel_size=3, stride=2, padding=1)

# Pass the input data through the convolutional layer
output_data = conv_layer(input_data)

import pdb; pdb.set_trace()

# Print the sizes of all the tensors
print(f"Input data size: {input_data.size()}")
print(f"Convolutional layer weight size: {conv_layer.weight.size()}")
print(f"Convolutional layer bias size: {conv_layer.bias.size()}")
print(f"Output data size: {output_data.size()}")

print(dir(conv_layer))

