import torch
import torch.nn as nn

class SquareRegularizeLoss(nn.Module):
	def __init__(self, p=1):
		super().__init__()
		self.p = p
	
	def forward(self, input):
		# print(input.size())
		# N = input.size(0)
		input = torch.pow(input, 2).sum(dim=1)
		if self.p == 1:
			loss = torch.abs(1-input)
		else:
			loss = torch.pow(1-input, self.p)
		loss = loss.mean()
		return loss