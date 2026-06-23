from torch import nn
import functools
class Discriminator(nn.Module):
    def __init__(self, config= None, num_filters_last=64, n_layers=3):
        super(Discriminator, self).__init__()

        layers = [nn.Conv2d(3, num_filters_last, 4, 2, 1), nn.LeakyReLU(0.2)]
        num_filters_mult = 1

        for i in range(1, n_layers + 1):
            num_filters_mult_last = num_filters_mult
            num_filters_mult = min(2 ** i, 8)
            layers += [
                nn.Conv2d(num_filters_last * num_filters_mult_last, num_filters_last * num_filters_mult, 4,
                          2 if i < n_layers else 1, 1, bias=False),
                nn.BatchNorm2d(num_filters_last * num_filters_mult),
                nn.LeakyReLU(0.2, True)
            ]

        layers.append(nn.Conv2d(num_filters_last * num_filters_mult, 1, 4, 1, 1))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

class Discriminator2(nn.Module):
    def __init__(self, config=None, num_filters_last=64, n_layers=3):
        super(Discriminator2, self).__init__()

        layers = [
            nn.ReflectionPad2d(1),
            nn.Conv2d(3, num_filters_last, 4, 2, 0),  # padding=0
            nn.LeakyReLU(0.2)
        ]

        num_filters_mult = 1
        for i in range(1, n_layers + 1):
            num_filters_mult_last = num_filters_mult
            num_filters_mult = min(2 ** i, 8)

            layers += [
                nn.ReflectionPad2d(1),
                nn.Conv2d(num_filters_last * num_filters_mult_last,
                          num_filters_last * num_filters_mult,
                          4, 2 if i < n_layers else 1, 0, bias=True),
                nn.LeakyReLU(0.2, True)
            ]

        layers += [
            nn.ReflectionPad2d(1),
            nn.utils.spectral_norm(
                nn.Conv2d(num_filters_last * num_filters_mult, 1, 4, 1, 0)
            )
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)