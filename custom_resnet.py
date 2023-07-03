import torch.nn as nn
import torchinfo


class ConvLayer(nn.Module):
    def __init__(self, input_c, output_c, bias=False, stride=1, padding=1, pool=False):
        super(ConvLayer, self).__init__()

        layers = list()
        layers.append(
            nn.Conv2d(input_c, output_c, 3, bias=bias, stride=stride, padding=padding, padding_mode='replicate')
        )
        if pool:
            layers.append(nn.MaxPool2d(2, 2))
        layers.append(nn.BatchNorm2d(output_c))
        layers.append(nn.ReLU())

        self.all_layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.all_layers(x)


class CustomLayer(nn.Module):
    def __init__(self, input_c, output_c, pool=True, residue=2):
        super(CustomLayer, self).__init__()

        self.pool_block = ConvLayer(input_c, output_c, pool=pool)
        self.res_block = None
        if residue > 0:
            layers = list()
            for i in range(0, residue):
                layers.append(
                    ConvLayer(output_c, output_c, pool=False)
                )
            self.res_block = nn.Sequential(*layers)

    def forward(self, x):
        x = self.pool_block(x)
        if self.res_block is not None:
            x_ = x
            x = self.res_block(x)
            # += operator causes inplace errors in pytorch if done right after relu.
            x = x + x_
        return x


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.all_layers = nn.Sequential(
            CustomLayer(3, 64, pool=False, residue=0),
            CustomLayer(64, 128, pool=True, residue=2),
            CustomLayer(128, 256, pool=True, residue=0),
            CustomLayer(256, 512, pool=True, residue=2),
            nn.MaxPool2d(4, 4),
            nn.Flatten(),
            nn.Linear(512, 10),
            # nn.Softmax()   #  This is not required as crossentropy combines logsoftmax and nllloss.
            # https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
        )

    def forward(self, x):
        return self.all_layers(x)

    def summary(self, input_size=None, depth=10):
        return torchinfo.summary(self, input_size=input_size, depth=depth,
                                 col_names=["input_size", "output_size", "num_params", "params_percent"])