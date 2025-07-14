from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import copy
import logging
import math
import torch
import torch.nn as nn
import numpy as np
from torch.nn import Dropout, Softmax, Conv2d, LayerNorm
from torch.nn.modules.utils import _pair

logger = logging.getLogger(__name__)


class Channel_Embeddings(nn.Module):
    """Constructs patch-based and position embeddings for input images."""
    def __init__(self, config, patchsize, img_size, in_channels):
        super().__init__()
        # Ensuring img_size and patch_size are tuples by applying _pair
        img_size = _pair(img_size)
        patch_size = _pair(patchsize)
        # TODO: Calculate the number of patches by dividing image dimensions by patch dimensions
        n_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])

        # Convolution for extracting patch embeddings
        self.patch_embeddings = Conv2d(in_channels=in_channels,
                                       out_channels=in_channels,
                                       kernel_size=patch_size,
                                       stride=patch_size)
        # TODO: Initialize position embeddings as a learnable parameter initialized with zeros of size 1 * number of patches * input channel
        self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches, in_channels))
        
        # Dropout layer with rate from config
        self.dropout = Dropout(config.transformer["embeddings_dropout_rate"])

    def forward(self, x):
        # Add condition to return None if input x is None
        if x is None:
            return None
        # Generate patch embeddings
        x = self.patch_embeddings(x)  # Shape (B, hidden, n_patches^(1/2), n_patches^(1/2))
        # TODO: Flatten the spatial dimensions and transpose to get shape (B, n_patches, hidden)
        x = x.flatten(2)
        x = x.transpose(-1, -2)
        # TODO: Add position embeddings to the flattened patches
        embeddings = x + self.position_embeddings
        # Applying dropout to the combined embeddings
        embeddings = self.dropout(embeddings)
        return embeddings


class Reconstruct(nn.Module):
    """Reconstructs feature map to original resolution using upsampling and convolution."""
    def __init__(self, in_channels, out_channels, kernel_size, scale_factor):
        super(Reconstruct, self).__init__()
        # Setting padding based on kernel size (1 for kernel size 3, else 0)
        padding = 1 if kernel_size == 3 else 0
        # Convolution to reconstruct feature maps
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
        # Batch normalization and activation
        self.norm = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU(inplace=True)
        self.scale_factor = scale_factor

    def forward(self, x):
        # Adding condition to return None if x is None
        if x is None:
            return None
        # Reshape and upsample the input tensor
        B, n_patch, hidden = x.size()  # Reshape from (B, n_patch, hidden) to (B, h, w, hidden)
        # TODO: Calculate the dimensions of height and width. they will be the sqrt of number of patches
        h, w = int(np.sqrt(n_patch)), int(np.sqrt(n_patch))
        # Permute and reshape x to match 2D convolution input format
        x = x.permute(0, 2, 1).contiguous().view(B, hidden, h, w)
        # Upsample x to scale factor
        x = nn.Upsample(scale_factor=self.scale_factor)(x)
        # Apply convolution, normalization, and activation
        out = self.conv(x)
        out = self.norm(out)
        out = self.activation(out)
        return out


class Attention_org(nn.Module):
    """Defines a multi-head attention mechanism with individual attention layers for multiple channels."""
    def __init__(self, config, vis, channel_num):
        super(Attention_org, self).__init__()
        self.vis = vis
        self.KV_size = config.KV_size
        self.channel_num = channel_num
        self.num_attention_heads = config.transformer["num_heads"]

        # TODO: Initialize module lists for query, key, and value linear layers using ModuleList class of nn
        self.query1 = nn.ModuleList()
        self.query2 = nn.ModuleList()
        self.query3 = nn.ModuleList()
        self.query4 = nn.ModuleList()
        self.key = nn.ModuleList()
        self.value = nn.ModuleList()

        # For each head, initialize query, key, and value layers
        for _ in range(self.num_attention_heads):

            # TODO: Define querys for each input as a linear layers with channel_num of that input as input and output sizes
            # False the biases
            query1 = nn.Linear(channel_num[0], channel_num[0], bias=False)
            query2 = nn.Linear(channel_num[1], channel_num[1], bias=False)
            query3 = nn.Linear(channel_num[2], channel_num[2], bias=False)
            query4 = nn.Linear(channel_num[3], channel_num[3], bias=False)
            
            # TODO: Define `key` and `value` as linear layers with self.KV_size as input and output sizes
            # False the biases
            key = nn.Linear( self.KV_size,  self.KV_size, bias=False)
            value = nn.Linear(self.KV_size,  self.KV_size, bias=False)

            # Append initialized layers to their respective lists
            self.query1.append(copy.deepcopy(query1))
            self.query2.append(copy.deepcopy(query2))
            self.query3.append(copy.deepcopy(query3))
            self.query4.append(copy.deepcopy(query4))
            self.key.append(copy.deepcopy(key))
            self.value.append(copy.deepcopy(value))
        
        # Initialize InstanceNorm, Softmax, Dropout layers for attention
        self.psi = nn.InstanceNorm2d(self.num_attention_heads)
        self.softmax = Softmax(dim=3)
        self.attn_dropout = Dropout(config.transformer["attention_dropout_rate"])
        self.proj_dropout = Dropout(config.transformer["attention_dropout_rate"])

        # Output linear layers for each channel
        self.out1 = nn.Linear(channel_num[0], channel_num[0], bias=False)
        self.out2 = nn.Linear(channel_num[1], channel_num[1], bias=False)
        self.out3 = nn.Linear(channel_num[2], channel_num[2], bias=False)
        self.out4 = nn.Linear(channel_num[3], channel_num[3], bias=False)

    def forward(self, emb1, emb2, emb3, emb4, emb_all):
        # Create empty lists for multi-head queries, keys, and values
        multi_head_Q1_list = []
        multi_head_Q2_list = []
        multi_head_Q3_list = []
        multi_head_Q4_list = []
        multi_head_K_list = []
        multi_head_V_list = []

        # Generate query lists for each head, append to multi_head_Q1_list, etc., if embeddings are not None
        if emb1 is not None:
            for query1 in self.query1:
                Q1 = query1(emb1)
                multi_head_Q1_list.append(Q1)
        
        # TODO: Repeat similar steps for emb2, emb3, emb4 to generate multi-head query lists
        if emb2 is not None:
            for query2 in self.query2:
                Q2 = query2(emb2)
                multi_head_Q2_list.append(Q2)
        if emb3 is not None:
            for query3 in self.query3:
                Q3 = query3(emb3)
                multi_head_Q3_list.append(Q3)
        if emb4 is not None:
            for query4 in self.query4:
                Q4 = query4(emb4)
                multi_head_Q4_list.append(Q4)
        
        # TODO: Generate keys and values for all heads based on emb_all
        for key in self.key:
            K = key(emb_all)
            multi_head_K_list.append(K)
        
        # TODO: Append each output of value layers to multi_head_V_list
        for value in self.value:
            V = value(emb_all)
            multi_head_V_list.append(V)

        # Stack query, key, and value lists to form multi-head tensors
        multi_head_Q1 = torch.stack(multi_head_Q1_list, dim=1) if emb1 is not None else None
        #TODO: Do the same for Q2, Q3, Q4
        multi_head_Q2 = torch.stack(multi_head_Q2_list, dim=1) if emb2 is not None else None
        multi_head_Q3 = torch.stack(multi_head_Q3_list, dim=1) if emb3 is not None else None
        multi_head_Q4 = torch.stack(multi_head_Q4_list, dim=1) if emb4 is not None else None
        
        multi_head_K = torch.stack(multi_head_K_list, dim=1)
        multi_head_V = torch.stack(multi_head_V_list, dim=1)

        # Transpose the multi_head_Q tensors for matrix multiplication compatibility
        multi_head_Q1 = multi_head_Q1.transpose(-1, -2) if emb1 is not None else None
        #TODO: Do the same for Q2, Q3, Q4
        multi_head_Q2 = multi_head_Q2.transpose(-1, -2) if emb2 is not None else None
        multi_head_Q3 = multi_head_Q3.transpose(-1, -2) if emb3 is not None else None
        multi_head_Q4 = multi_head_Q4.transpose(-1, -2) if emb4 is not None else None
        # Compute attention scores by performing matrix multiplication with queries and keys
        attention_scores1 = torch.matmul(multi_head_Q1, multi_head_K) if emb1 is not None else None
        #TODO: Do the same for Q2, Q3, Q4
        attention_scores2 = torch.matmul(multi_head_Q2, multi_head_K) if emb2 is not None else None
        attention_scores3 = torch.matmul(multi_head_Q3, multi_head_K) if emb3 is not None else None
        attention_scores4 = torch.matmul(multi_head_Q4, multi_head_K) if emb4 is not None else None

        # Scale the attention scores by the square root of self.KV_size
        attention_scores1 = attention_scores1 / math.sqrt(self.KV_size) if emb1 is not None else None
        #TODO: Do the same for Q2, Q3, Q4
        attention_scores2 = attention_scores2 / math.sqrt(self.KV_size) if emb2 is not None else None
        attention_scores3 = attention_scores3 / math.sqrt(self.KV_size) if emb3 is not None else None
        attention_scores4 = attention_scores4 / math.sqrt(self.KV_size) if emb4 is not None else None

        # Apply softmax to obtain attention probabilities
        attention_probs1 = self.softmax(self.psi(attention_scores1)) if emb1 is not None else None
        #TODO: Do the same for Q2, Q3, Q4
        attention_probs2 = self.softmax(self.psi(attention_scores2)) if emb2 is not None else None
        attention_probs3 = self.softmax(self.psi(attention_scores3)) if emb3 is not None else None
        attention_probs4 = self.softmax(self.psi(attention_scores4)) if emb4 is not None else None

        if self.vis:
            weights =  []
            weights.append(attention_probs1.mean(1))
            weights.append(attention_probs2.mean(1))
            weights.append(attention_probs3.mean(1))
            weights.append(attention_probs4.mean(1))
        else: weights=None

        # Apply dropout to the attention probabilities if emb1 is not None
        attention_probs1 = self.attn_dropout(attention_probs1) if emb1 is not None else None
        #TODO: Do the same for Q2, Q3, Q4
        attention_probs2 = self.attn_dropout(attention_probs2) if emb2 is not None else None
        attention_probs3 = self.attn_dropout(attention_probs3) if emb3 is not None else None
        attention_probs4 = self.attn_dropout(attention_probs4) if emb4 is not None else None
        multi_head_V = multi_head_V.transpose(-1, -2)
        
        # Obtain the final context layers by multiplying attention_probs with values
        context_layer1 = torch.matmul(attention_probs1, multi_head_V) if emb1 is not None else None
        #TODO: Do the same for Q2, Q3, Q4
        context_layer2 = torch.matmul(attention_probs2, multi_head_V) if emb2 is not None else None
        context_layer3 = torch.matmul(attention_probs3, multi_head_V) if emb3 is not None else None
        context_layer4 = torch.matmul(attention_probs4, multi_head_V) if emb4 is not None else None

        # Permute and reshape context layers back to the original format
        context_layer1 = context_layer1.permute(0, 3, 2, 1).contiguous() if emb1 is not None else None
        context_layer1 = context_layer1.mean(dim=3) if emb1 is not None else None
        #TODO: Do the same for Q2, Q3, Q4
        context_layer2 = context_layer2.permute(0, 3, 2, 1).contiguous() if emb2 is not None else None
        context_layer2 = context_layer2.mean(dim=3) if emb2 is not None else None
        context_layer3 = context_layer3.permute(0, 3, 2, 1).contiguous() if emb3 is not None else None
        context_layer3 = context_layer3.mean(dim=3) if emb3 is not None else None
        context_layer4 = context_layer4.permute(0, 3, 2, 1).contiguous() if emb4 is not None else None
        context_layer4 = context_layer4.mean(dim=3) if emb4 is not None else None

        # Apply output linear transformations and dropout
        O1 = self.out1(context_layer1) if emb1 is not None else None
        O1 = self.proj_dropout(O1) if emb1 is not None else None
        #TODO: Do the same for Q2, Q3, Q4
        O2 = self.out2(context_layer2) if emb2 is not None else None
        O2 = self.proj_dropout(O2) if emb2 is not None else None
        O3 = self.out3(context_layer3) if emb3 is not None else None
        O3 = self.proj_dropout(O3) if emb3 is not None else None
        O4 = self.out4(context_layer4) if emb4 is not None else None
        O4 = self.proj_dropout(O4) if emb4 is not None else None
        # Return the outputs and attention weights for visualization if required
        return O1, O2, O3, O4, weights


class Mlp(nn.Module):
    """Defines a Multi-Layer Perceptron (MLP) with two fully connected layers and GELU activation."""
    def __init__(self, config, in_channel, mlp_channel):
        super(Mlp, self).__init__()
        # TODO: Define the first fully connected layer (fc1) with input size in_channel and output size mlp_channel
        self.fc1 = nn.Linear(in_channel, mlp_channel)
        # TODO: Define the second fully connected layer (fc2) with input size mlp_channel and output size in_channel
        self.fc2 = nn.Linear(mlp_channel, in_channel)
        # Activation function
        self.act_fn = nn.GELU()
        # Dropout layer with rate from config
        self.dropout = Dropout(config.transformer["dropout_rate"])
        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initializes weights for fully connected layers."""
        # TODO: Use Xavier uniform initialization for fc1 weights
        nn.init.xavier_uniform_(self.fc1.weight)
        # TODO: Use Xavier uniform initialization for fc2 weights
        nn.init.xavier_uniform_(self.fc2.weight)
        # TODO: Initialize biases for fc1 and fc2 with normal distribution (mean=0, std=1e-6)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        # TODO: Apply first fully connected layer (fc1) to the input x
        x = self.fc1(x)
        # TODO: Apply activation function (GELU)
        x = self.act_fn(x)
        # TODO: Apply dropout after the activation function
        x = self.dropout(x)
        # TODO: Apply the second fully connected layer (fc2) to the transformed input
        x = self.fc2(x)
        # TODO: Apply dropout again after the second layer
        x = self.dropout(x)
        # Return the output
        return x

class Block_ViT(nn.Module):
    """Defines a Vision Transformer (ViT) block with attention normalization, channel attention, and feed-forward layers."""
    def __init__(self, config, vis, channel_num):
        super(Block_ViT, self).__init__()
        # Expansion ratio for the hidden size in the feed-forward network
        expand_ratio = config.expand_ratio
        # Define LayerNorm layers for attention normalization for each input channel, with eps=1e-6
        self.attn_norm1 = LayerNorm(channel_num[0], eps=1e-6)
        self.attn_norm2 = LayerNorm(channel_num[1], eps=1e-6)
        self.attn_norm3 = LayerNorm(channel_num[2], eps=1e-6)
        self.attn_norm4 = LayerNorm(channel_num[3], eps=1e-6)
        # Define LayerNorm for the concatenated channel embeddings using config.KV_size as dimension
        self.attn_norm = LayerNorm(config.KV_size, eps=1e-6)
        # TODO: Channel-wise attention mechanism using Attention_org class with proper arguments
        self.channel_attn = Attention_org(config, vis, channel_num)
        # Define LayerNorm layers for feed-forward normalization for each input channel
        self.ffn_norm1 = LayerNorm(channel_num[0], eps=1e-6)
        self.ffn_norm2 = LayerNorm(channel_num[1], eps=1e-6)
        self.ffn_norm3 = LayerNorm(channel_num[2], eps=1e-6)
        self.ffn_norm4 = LayerNorm(channel_num[3], eps=1e-6)
        # Define feed-forward network (MLP) layers for each channel with the specified expansion ratio
        self.ffn1 = Mlp(config, channel_num[0], channel_num[0] * expand_ratio)
        # TODO: do the same for ffn2, ffn3, ffn4
        self.ffn2 = Mlp(config,channel_num[1],channel_num[1]*expand_ratio)
        self.ffn3 = Mlp(config,channel_num[2],channel_num[2]*expand_ratio)
        self.ffn4 = Mlp(config,channel_num[3],channel_num[3]*expand_ratio)
        
        

    def forward(self, emb1, emb2, emb3, emb4):
        embcat = []

        # Store original embeddings for residual (skip) connections
        org1, org2, org3, org4 = emb1, emb2, emb3, emb4
        
        # Append each embedding to embcat if it is not None to form the concatenated input
        for i in range(4):
            var_name = "emb" + str(i + 1)
            tmp_var = locals()[var_name]
            if tmp_var is not None:
                embcat.append(tmp_var)

        # Concatenate embeddings along the channel dimension
        emb_all = torch.cat(embcat, dim=2)

        # Normalize each individual embedding using the corresponding attention normalization layer
        cx1 = self.attn_norm1(emb1) if emb1 is not None else None
        # TODO: do the same for cx2, cx3, cx4
        cx2 = self.attn_norm2(emb2) if emb2 is not None else None
        cx3 = self.attn_norm3(emb3) if emb3 is not None else None
        cx4 = self.attn_norm4(emb4) if emb4 is not None else None
        

        # Apply normalization to the concatenated embeddings
        emb_all = self.attn_norm(emb_all)

        # TODO: Pass normalized embeddings to the channel attention and obtain weighted outputs
        cx1, cx2, cx3, cx4, weights = self.channel_attn(cx1,cx2,cx3,cx4,emb_all)

        # TODO: Add the original embeddings (org1-org4) back to each attention output to form residual connections
        cx1 = org1 + cx1 if emb1 is not None else None
        cx2 = org2 + cx2 if emb2 is not None else None
        cx3 = org3 + cx3 if emb3 is not None else None
        cx4 = org4 + cx4 if emb4 is not None else None

        # Update the original embeddings after adding attention outputs for feed-forward processing
        org1, org2, org3, org4 = cx1, cx2, cx3, cx4

        # TODO: Apply feed-forward normalization and MLP transformations to each attention output
        x1 = self.ffn_norm1(cx1) if emb1 is not None else None
        x2 = self.ffn_norm2(cx2) if emb2 is not None else None
        x3 = self.ffn_norm3(cx3) if emb3 is not None else None
        x4 = self.ffn_norm4(cx4) if emb4 is not None else None

        x1 = self.ffn1(x1) if emb1 is not None else None
        x2 = self.ffn2(x2) if emb2 is not None else None
        x3 = self.ffn3(x3) if emb3 is not None else None
        x4 = self.ffn4(x4) if emb4 is not None else None

        # TODO: Add original embeddings back to the transformed outputs to form residual connections
        x1 = x1 + org1 if emb1 is not None else None
        x2 = x2 + org2 if emb2 is not None else None
        x3 = x3 + org3 if emb3 is not None else None
        x4 = x4 + org4 if emb4 is not None else None

        # Return the final outputs and the attention weights
        return x1, x2, x3, x4, weights


class Encoder(nn.Module):
    """Defines an encoder with multiple ViT blocks and layer normalization for each input channel."""
    def __init__(self, config, vis, channel_num):
        super(Encoder, self).__init__()
        self.vis = vis
        # Initialize a ModuleList to hold the sequence of Block_ViT layers
        self.layer = nn.ModuleList()
        # Define LayerNorm for each input channel, with eps=1e-6 for numerical stability
        self.encoder_norm1 = LayerNorm(channel_num[0], eps=1e-6)
        self.encoder_norm2 = LayerNorm(channel_num[1], eps=1e-6)
        self.encoder_norm3 = LayerNorm(channel_num[2], eps=1e-6)
        self.encoder_norm4 = LayerNorm(channel_num[3], eps=1e-6)
        # Add a specified number of Block_ViT layers to the encoder
        for _ in range(config.transformer["num_layers"]):
            # TODO: Initialize a Block_ViT layer with the given config, vis, and channel_num
            layer = Block_ViT(config, vis, channel_num)
            # TODO: Append a deep copy of the initialized Block_ViT layer to self.layer
            self.layer.append(copy.deepcopy(layer))

    def forward(self, emb1, emb2, emb3, emb4):
        # Initialize a list to hold attention weights if visualization is enabled
        attn_weights = []
        # TODO: Pass each embedding (emb1-emb4) through the sequence of Block_ViT layers
        for layer_block in self.layer:
            # The Block_ViT layer processes all four embeddings and returns updated embeddings and attention weights
            emb1, emb2, emb3, emb4, weights = layer_block(emb1,emb2,emb3,emb4)
            # Append attention weights if visualization (vis) is enabled
            if self.vis:
                attn_weights.append(weights)
        # TODO: Apply LayerNorm to the final outputs of each embedding channel
        emb1 = self.encoder_norm1(emb1) if emb1 is not None else None
        emb2 = self.encoder_norm2(emb2) if emb2 is not None else None
        emb3 = self.encoder_norm3(emb3) if emb3 is not None else None
        emb4 = self.encoder_norm4(emb4) if emb4 is not None else None
        # Return the normalized embeddings and collected attention weights
        return emb1, emb2, emb3, emb4, attn_weights


class ChannelTransformer(nn.Module):
    """Combines patch embeddings, an encoder, and reconstruction layers for a complete channel transformer model."""
    def __init__(self, config, vis, img_size, channel_num=[64, 128, 256, 512], patchSize=[32, 16, 8, 4]):
        super().__init__()

        # Define patch sizes for each embedding layer using elements of patchSize
        self.patchSize_1 = patchSize[0]
        self.patchSize_2 = patchSize[1]
        self.patchSize_3 = patchSize[2]
        self.patchSize_4 = patchSize[3]

        # TODO: Define Channel_Embeddings for each channel with specified patch size and image size
        self.embeddings_1 = Channel_Embeddings(config, self.patchSize_1, img_size=img_size, in_channels=channel_num[0])
        self.embeddings_2 = Channel_Embeddings(config,self.patchSize_2, img_size=img_size//2, in_channels=channel_num[1])
        self.embeddings_3 = Channel_Embeddings(config,self.patchSize_3, img_size=img_size//4, in_channels=channel_num[2])
        self.embeddings_4 = Channel_Embeddings(config,self.patchSize_4, img_size=img_size//8, in_channels=channel_num[3])

        # TODO: Initialize the Encoder with config, vis flag, and channel_num list
        self.encoder = Encoder(config, vis, channel_num)

        # TODO: Define reconstruction layers to upsample back to the original feature map dimensions
        self.reconstruct_1 = Reconstruct(channel_num[0], channel_num[0], kernel_size=1, scale_factor=(self.patchSize_1, self.patchSize_1))
        self.reconstruct_2 = Reconstruct(channel_num[1], channel_num[1], kernel_size=1,scale_factor=(self.patchSize_2,self.patchSize_2))
        self.reconstruct_3 = Reconstruct(channel_num[2], channel_num[2], kernel_size=1,scale_factor=(self.patchSize_3,self.patchSize_3))
        self.reconstruct_4 = Reconstruct(channel_num[3], channel_num[3], kernel_size=1,scale_factor=(self.patchSize_4,self.patchSize_4))

    def forward(self, en1, en2, en3, en4):
        # TODO: Create embeddings from input feature maps en1, en2, en3, and en4
        emb1 = self.embeddings_1(en1)
        emb2 = self.embeddings_2(en2)
        emb3 = self.embeddings_3(en3)
        emb4 = self.embeddings_4(en4)

        # TODO: Pass embeddings through the encoder and obtain encoded representations and attention weights
        encoded1, encoded2, encoded3, encoded4, attn_weights = self.encoder(emb1,emb2,emb3,emb4)

        # TODO: Reconstruct each encoding to its original spatial resolution using corresponding reconstruct layers
        x1 = self.reconstruct_1(encoded1) if en1 is not None else None
        x2 = self.reconstruct_2(encoded2) if en2 is not None else None
        x3 = self.reconstruct_3(encoded3) if en3 is not None else None
        x4 = self.reconstruct_4(encoded4) if en4 is not None else None

        # TODO: Add the reconstructed outputs back to the original inputs (residual connections)
        x1 = x1 + en1  if en1 is not None else None
        x2 = x2 + en2  if en2 is not None else None
        x3 = x3 + en3  if en3 is not None else None
        x4 = x4 + en4  if en4 is not None else None

        # Return the final outputs and attention weights for visualization if needed
        return x1, x2, x3, x4, attn_weights
