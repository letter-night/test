import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from math import sqrt
from positional_encodings.torch_encodings import PositionalEncoding2D

import pytorch_lightning as pl

from torch import distributions as dist
import torch.optim as optim 

import torch.distributions as D 



def magnitude_max_pooling_1d(input_tensor, pool_size, stride):
	# Get the dimensions of the input tensor
	B, N, L = input_tensor.size()

	# Calculate the output length
	out_length = (L - pool_size) // stride + 1

	# Unfold the input tensor to create sliding windows
	windows = input_tensor.unfold(2, pool_size, stride)

	# Reshape the windows to a 4D tensor
	windows = windows.contiguous().view(B, N, out_length, pool_size)

	# Compute the magnitudes of the values in each window
	magnitudes = torch.abs(windows)

	# Find the indices of the maximum magnitudes in each window
	max_indices = torch.argmax(magnitudes, dim=-1, keepdim=True)

	# Gather the values corresponding to the maximum magnitudes
	max_values = windows.gather(dim=-1, index=max_indices).squeeze(-1)

	return max_values


class TriangularCausalMask():
	def __init__(self, B, L, device="cpu"):
		mask_shape = [B, 1, L, L]
		with torch.no_grad():
			self._mask = torch.triu(torch.ones(mask_shape, dtype=torch.bool), diagonal=1).to(device)
	
	@property
	def mask(self):
		return self._mask



class DataEmbedding_FeaturePatching(nn.Module):
	def __init__(self, seq_len, patch_size, embed_dim=512, dropout=0.1):
		super(DataEmbedding_FeaturePatching, self).__init__()
		self.seq_len=seq_len
		self.patch_size=patch_size
		self.n_of_patches = (seq_len - patch_size) // (patch_size//2) + 1
		self.inner_dim = patch_size * 10
		self.embed_dim = embed_dim

		self.conv1 = nn.Conv1d(1, 3, kernel_size=5)
		self.conv2 = nn.Conv1d(1, 3, kernel_size=9)
		self.conv3 = nn.Conv1d(1, 3, kernel_size=15)
		self.gelu1 = nn.GELU()
		self.gelu2 = nn.GELU()
		self.fc1 = nn.Linear(self.inner_dim, embed_dim*4)
		self.fc2 = nn.Linear(embed_dim*4, embed_dim)
		self.pe = PositionalEncoding2D(embed_dim)
		self.dropout = nn.Dropout(p=dropout)

		self.sigm = nn.GELU()
	
	def forward(self, x):
		B, L, N = x.shape
		x = x.permute(0, 2, 1)
		# x: [Batch Variate Time]

		x = x.reshape(-1, 1, L)
		x_1 = F.pad(x, (4, 0), mode='replicate')
		x_1 = self.conv1(x_1)
		x_2 = F.pad(x, (8, 0), mode='replicate')
		x_2 = self.conv2(x_2)
		x_3 = F.pad(x, (14, 0), mode='replicate')
		x_3 = self.conv3(x_3)
		x_1 = F.pad(x_1, (2, 0), mode='constant', value=0)
		x_2 = F.pad(x_2, (4, 0), mode='constant', value=0)
		x_3 = F.pad(x_3, (6, 0), mode='constant', value=0)

		x_1 = magnitude_max_pooling_1d(x_1, 3, 1)
		x_2 = magnitude_max_pooling_1d(x_2, 5, 1)
		x_3 = magnitude_max_pooling_1d(x_3, 7, 1)

		x_1 = x_1.reshape(B, N, 3, L)
		x_2 = x_2.reshape(B, N, 3, L)
		x_3 = x_3.reshape(B, N, 3, L)
		x = x.reshape(B, N, 1, L)

		x_1 = x_1.unfold(3, self.patch_size, self.patch_size//2)
		x_2 = x_2.unfold(3, self.patch_size, self.patch_size//2)
		x_3 = x_3.unfold(3, self.patch_size, self.patch_size//2)
		x = x.unfold(3, self.patch_size, self.patch_size//2)

		# B, N, num_patches, 3, patch_size
		x_1 = x_1.permute(0, 1, 3, 2, 4)
		x_2 = x_2.permute(0, 1, 3, 2, 4)
		x_3 = x_3.permute(0, 1, 3, 2, 4)
		x = x.permute(0, 1, 3, 2, 4)

		x = torch.cat([x, x_1, x_2, x_3], dim=3)
		# B, N, num_patches, num_channels (3+3+3+1), patch_size

		x = x.reshape(B, N, self.n_of_patches, -1)
		x = self.gelu1(self.fc1(x))
		x = self.fc2(x)
		x = self.pe(x) + x

		x = x.reshape(B, -1, self.embed_dim)

		return self.dropout(x)



class FullAttention(nn.Module):
	def __init__(self, mask_flag=True, scale=None, attention_dropout=0.1, output_attention=False):
		super(FullAttention, self).__init__()
		self.scale=scale
		self.mask_flag=mask_flag
		self.output_attention=output_attention
		self.dropout=nn.Dropout(attention_dropout)
	
	def forward(self, queries, keys, values, attn_mask):
		B, L, H, E = queries.shape
		_, S, _, D = values.shape
		scale = self.scale or 1. / sqrt(E)

		scores = torch.einsum("blhe,bshe->bhls", queries, keys)

		if self.mask_flag:
			if attn_mask is None:
				attn_mask = TriangularCausalMask(B, L, device=queries.device)
			
			scores.masked_fill_(attn_mask.mask, -np.inf)
		
		A = self.dropout(torch.softmax(scale * scores, dim=-1))
		V = torch.einsum("bhls,bshd->blhd", A, values)

		if self.output_attention:
			return (V.contiguous(), A)
		else:
			return (V.contiguous(), None)



class AttentionLayer(nn.Module):
	def __init__(self, attention, d_model, n_heads, d_keys=None, d_values=None):
		super(AttentionLayer, self).__init__()

		d_keys=d_keys or (d_model // n_heads)
		d_values = d_values or (d_model // n_heads)

		self.inner_attention=attention
		self.query_projection = nn.Linear(d_model, d_keys * n_heads)
		self.key_projection = nn.Linear(d_model, d_keys * n_heads)
		self.value_projection = nn.Linear(d_model, d_values * n_heads)
		self.output_projection = nn.Linear(d_values * n_heads, d_model)
		self.n_heads=n_heads
	
	def forward(self, queries, keys, values, attn_mask):
		B, L, _ = queries.shape
		_, S, _ = keys.shape
		H = self.n_heads

		queries = self.query_projection(queries).view(B, L, H, -1)
		keys = self.key_projection(keys).view(B, S, H, -1)
		values = self.value_projection(values).view(B, S, H, -1)

		out, attn = self.inner_attention(
			queries, keys, values, attn_mask
		)
		out = out.view(B, L, -1)

		return self.output_projection(out), attn



class EncoderLayer(nn.Module):
	def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="silu"):
		super(EncoderLayer, self).__init__()
		d_ff=d_ff or 4 * d_model
		self.attention=attention
		self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
		self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
		self.norm1 = nn.LayerNorm(d_model)
		self.norm2 = nn.LayerNorm(d_model)
		self.dropout = nn.Dropout(dropout)
		self.activation = F.gelu if activation == "gelu" else F.silu
	
	def forward(self, x, attn_mask=None):
		new_x, attn = self.attention(
			x, x, x, attn_mask=attn_mask,
		)
		x = x + self.dropout(new_x)

		y = x = self.norm1(x)
		y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
		y = self.dropout(self.conv2(y).transpose(-1, 1))

		return self.norm2(x + y), attn
	


class Encoder(nn.Module):
	def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
		super(Encoder, self).__init__()
		self.attn_layers = nn.ModuleList(attn_layers)
		self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
		self.norm = norm_layer
	
	def forward(self, x, attn_mask=None):
		# x [B, L, D]
		attns = []
		if self.conv_layers is not None:
			for i, (attn_layer, conv_layer) in enumerate(zip(self.attn_layers, self.conv_layers)):
				out, attn = attn_layer(x, attn_mask=attn_mask)
				x = x.unsqueeze(1).permute(0, 2, 3, 1)
				x = conv_layer(x)
				x = x.squeeze(3)
				attns.append(attn)
			x, attn = self.attn_layers[-1](x)
			attns.append(attn)
		else:
			for attn_layer in self.attn_layers:
				x, attn = attn_layer(x, attn_mask=attn_mask)
				attns.append(attn)
		
		if self.norm is not None:
			x = self.norm(x)
		
		return x, attns