import torch
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self, num_classes=39, input_shape=(3, 224, 224)):
        super().__init__()
        
        layers = []
        in_channels = input_shape[0]
        
        channel_configs = [32, 64, 128, 256]
        
        for out_channels in channel_configs:
            layers.extend(self._conv_block(in_channels, out_channels))
            in_channels = out_channels
            
        self.conv_layers = nn.Sequential(*layers)

        self._calculate_linear_input(input_shape)
        
        self.dense_layers = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(self.linear_input_size, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(1024, num_classes),
        )

    def _conv_block(self, in_c, out_c):
        return [
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_c),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_c),
            nn.MaxPool2d(kernel_size=2)
        ]
        
    def _calculate_linear_input(self, input_shape):
        with torch.no_grad():
            dummy_input = torch.zeros(1, *input_shape)
            dummy_output = self.conv_layers(dummy_input)
            self.linear_input_size = dummy_output.flatten(1).shape[1]

    def forward(self, x):
        out = self.conv_layers(x)
        out = torch.flatten(out, 1)
        out = self.dense_layers(out)
        return out
