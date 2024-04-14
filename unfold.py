import torch
import torch.nn as nn

fold = nn.Fold(output_size=(14, 14), kernel_size=(3, 3), stride=2)
input = torch.randn(2, 5 * 3 * 3, 36)
output = fold(input)
print (output.size())

