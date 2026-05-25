import torch
from torch.autograd import Function

class GRLFunction(Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input.view_as(input)

    @staticmethod
    def backward(ctx, grad_output):
        return -grad_output

class GradientReversal(torch.nn.Module):
    def forward(self, input):
        return GRLFunction.apply(input)