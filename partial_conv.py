import torch
import torch.nn.functional as F
from torch import nn, cuda
from torch.autograd import Variable

class PartialConv(nn.Conv2d):
    def __init__(self, *args, **kwargs):

        if 'multi_channel' in kwargs:
            self.multi_channel = kwargs['multi_channel']
            kwargs.pop('multi_channel')
        else:
            self.multi_channel = False  

        if 'return_mask' in kwargs:
            self.return_mask = kwargs['return_mask']
            kwargs.pop('return_mask')
        else:
            self.return_mask = True

        super(PartialConv, self).__init__(*args, **kwargs)

        if self.multi_channel:
            self.ones = torch.ones(self.out_channels, self.in_channels, self.kernel_size[0], self.kernel_size[1])
        else:
            self.ones = torch.ones(1, 1, self.kernel_size[0], self.kernel_size[1])

        self.slide_winsize = self.ones.shape[1] * self.ones.shape[2] * self.ones.shape[3]

        self.last_size = (None, None, None, None)

    def forward(self, input, mask_in):
        assert len(input.shape) == 4

        with torch.no_grad():
            if self.ones.type() != mask_in.type():
                self.ones = self.ones.to(mask_in)

            mask_out = F.conv2d(mask_in, self.ones, bias = None, stride = self.stride, padding = self.padding, dilation = self.dilation)

            multiplier = self.slide_winsize / (mask_out + 1e-6)

            mask_out = torch.clamp(mask_out, 0, 1)
            multiplier = torch.mul(multiplier, mask_out).to(input)

        raw_out = super(PartialConv, self).forward(torch.mul(input, mask_in.to(input)))

        if self.bias is not None:
            bias_view = self.bias.view(1, self.out_channels, 1, 1)
            output = torch.mul(raw_out - bias_view, multiplier) + bias_view
            output = torch.mul(output, mask_out)
        else:
            output = torch.mul(raw_out, multiplier)

        if self.return_mask:
            return output, mask_out
        else:
            return output