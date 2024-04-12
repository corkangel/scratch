import torch
import torch.nn as nn

# Define the batch size, input channels, output features, stride and padding
batch_size = 64
input_channels = 4
output_features = 8
stride = 2
padding = 1

# Create a random tensor to represent your input data
input_data = torch.randn(batch_size, input_channels, 14, 14)

# Define the convolutional layer
conv_layer = nn.Conv2d(input_channels, output_features, kernel_size=3, stride=stride, padding=padding)

# Pass the input data through the convolutional layer
output_data = conv_layer(input_data)

# Print the sizes of all the tensors
print(f"Input data size: {input_data.size()}")
print(f"Convolutional layer weight size: {conv_layer.weight.size()}")
print(f"Convolutional layer bias size: {conv_layer.bias.size()}")
print(f"Output data size: {output_data.size()}")
