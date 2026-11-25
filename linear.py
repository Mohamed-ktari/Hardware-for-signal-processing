import torch
import torch.nn as nn
import custom_cuda
class LinearFunction(torch.autograd.Function):
  @staticmethod
  def forward(ctx, X, W, b):
    # Save tensors needed for backward
    ctx.save_for_backward(X, W)
    # Call custom CUDA forward
    Y = custom_cuda.forward(X, W, b)
    return Y
  @staticmethod
  def backward(ctx, grad_output):
    X, W = ctx.saved_tensors
    # Call custom CUDA backward
    dX, dW, db = custom_cuda.backward(grad_output.contiguous(),X,W)
    return dX, dW, db