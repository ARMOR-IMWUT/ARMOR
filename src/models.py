from torch import nn
from torch.autograd.grad_mode import F


class CNNMnist(nn.Module):
    def __init__(self, args, useGAN=False, target_label=0):
        super(CNNMnist, self).__init__()
        self.conv1 = nn.Conv2d(args.num_channels, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        if useGAN:
            self.fc2 = nn.Linear(50, args.num_classes + 1)
            args.num_classes = args.num_classes + 1
            self.args = args
            args.num_classes = args.num_classes - 1
        else:
            self.fc2 = nn.Linear(50, args.num_classes)
            self.args = args
        self.useGAN = useGAN
        self.target_label = target_label

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, x.shape[1] * x.shape[2] * x.shape[3])
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        out = F.log_softmax(x, dim=1)
        return out


class CNNFashion_Mnist(nn.Module):
    def __init__(self, args, useGAN=False, target_label=0):
        super(CNNFashion_Mnist, self).__init__()

        # Convolution 1
        self.cnn1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=0)
        self.relu1 = nn.ReLU()
        # Max pool 1
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)

        # Convolution 2
        self.cnn2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=0)
        self.relu2 = nn.ReLU()

        # Max pool 2
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)

        self.dropout = nn.Dropout(p=0.5)

        if (useGAN):
            # Fully connected 1 (readout)
            self.fc1 = nn.Linear(32 * 4 * 4, args.num_classes + 1)
            args.num_classes = args.num_classes + 1
            self.args = args
            args.num_classes = args.num_classes - 1
        else:
            self.fc1 = nn.Linear(32 * 4 * 4, args.num_classes)
            self.args = args

        self.useGAN = useGAN
        self.target_label = target_label

    def forward(self, x):
        # Convolution 1
        out = self.cnn1(x)
        out = self.relu1(out)

        # Max pool 1
        out = self.maxpool1(out)

        # Convolution 2
        out = self.cnn2(out)
        out = self.relu2(out)

        # Max pool 2
        out = self.maxpool2(out)

        # Resize
        # Original size: (100, 32, 7, 7)
        # out.size(0): 100
        # New out size: (100, 32*7*7)
        out = out.view(out.size(0), -1)
        out = self.dropout(out)
        # Linear function (readout)
        out = self.fc1(out)

        return F.log_softmax(out, dim=1)


class CNNCifar:
    pass
