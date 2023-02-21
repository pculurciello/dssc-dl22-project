import torch
from torch import nn
import numpy as np
import itertools

class VAE(nn.Module):

    def __init__(self,
                 input_shape,
                 conv_filters,
                 conv_kernels,
                 conv_strides,
                 latent_space_dim):
        
        super().__init__()
        
        #shape of the tensor input to the first conv layer
        self.input_shape = input_shape # [1, N_MELS, n_hops]
        #shape of the input tensor at each encoder convolutional layer
        self.encoder_shape = []
        #shape of the input tensor at each decoder transpose convolutional layer
        self.decoder_shape = []
        self.conv_filters = conv_filters 
        self.conv_transpose_filters = conv_filters[:-1][::-1]
        self.conv_transpose_filters.append(1)
        self.conv_kernels = conv_kernels 
        self.conv_strides = conv_strides 
        self.shape_before_bottleneck = None
        self.latent_space_dim = latent_space_dim 
        self._num_conv_layers = len(conv_filters)
        self.enc_input_channels = self.conv_filters[:-1]
        self.enc_input_channels.insert(0, self.input_shape[0])
        self.enc_output_channels = self.conv_filters
        self.dec_input_channels = self.enc_output_channels[::-1]
        self.dec_output_channels = self.enc_input_channels[::-1]
        self.encoder = nn.Sequential()
        self.decoder = nn.Sequential()
        self.layer2id = dict()
        self.layer2id['encoder'] = dict()
        self.layer2id['decoder'] = dict()   
     
        self._compute_output_shape()
        self._build_encoder()
        self._compute_output_shape(encoder=False)
        self._build_decoder()
        #self._init_weights()
        #self._reset_params()
    
    # reset_params() and init_weights() are two methods for initializing model parameters
    def _reset_params(self):   
        for m in self._get_encoder_layers():
            reset_parameters = getattr(m, "reset_parameters", None)
            if callable(reset_parameters):
                m.reset_parameters()
        for m in self._get_decoder_layers():
            reset_parameters = getattr(m, "reset_parameters", None)
            if callable(reset_parameters):
                m.reset_parameters()
        
    def _init_weights(self):
        for m in self._get_encoder_layers():
            if (isinstance(m, nn.Conv2d) or 
                isinstance(m, nn.ConvTranspose2d) or 
                isinstance(m, nn.BatchNorm2d) or 
                isinstance(m, nn.Linear)):
                m.weight.data.normal_(mean=0.0, std=1.0)
                #print("normalizing layer " + str(m))
                if m.bias is not None:
                    m.bias.data.zero_()
        for m in self._get_decoder_layers():
            if (isinstance(m, nn.Conv2d) or 
                isinstance(m, nn.ConvTranspose2d) or 
                isinstance(m, nn.BatchNorm2d) or 
                isinstance(m, nn.Linear)):
                m.weight.data.normal_(mean=0.0, std=1.0)
                if m.bias is not None:
                    m.bias.data.zero_()
    
    def _build_encoder(self):
        self._add_conv_layers()
        self._add_bottleneck()

    def _build_decoder(self):
        self._invert_bottleneck()
        self._add_conv_transpose_layers()
            
        
    def _add_conv_layers(self):
        """Create all convolutional blocks in encoder."""
        for block_index in range(self._num_conv_layers):
            block_number = block_index + 1
            layer_index = 0
            self.layer2id['encoder']['conv{}'.format(block_number)] = 3 * block_index + layer_index 
            self.encoder.add_module(
                'conv{}'.format(block_number),
                self._add_conv_layer(block_index))
            layer_index += 1
            self.layer2id['encoder']['relu{}'.format(block_number)] = 3 * block_index + layer_index 
            self.encoder.add_module(
                'relu{}'.format(block_number),
                nn.ReLU())
            layer_index += 1
            self.layer2id['encoder']['batchnorm{}'.format(block_number)] = 3 * block_index + layer_index 
            self.encoder.add_module(
                'batchnorm{}'.format(block_number),
                nn.BatchNorm2d(self.conv_filters[block_index]))
            
    def _add_conv_layer(self, block_index):
        """Add a convolutional block to a graph of layers, consisting of
        conv 2d + ReLU + batch normalization.
        """
        conv_layer = nn.Conv2d(
            self.enc_input_channels[block_index],
            self.enc_output_channels[block_index],
            self.conv_kernels[block_index],
            stride=self.conv_strides[block_index],
            padding=1
        )
        return conv_layer
    
    def _compute_output_shape(self, encoder=True):
        #convolution output shape is computed with this formula (for each dimension): [(W−K+2P)/S]+1
        # we assume padding = (1, 1) and for transpose conv no output_padding and no dilation
        #W is the input volume
        #K is the Kernel size
        #P is the padding
        #S is the stride
        if encoder:
            inp = self.input_shape
            kernels = self.conv_kernels
            strides = self.conv_strides
            filters = self.conv_filters
        else:
            inp = self.encoder_shape[self._num_conv_layers - 1]
            kernels = self.conv_kernels[::-1]
            strides = self.conv_strides[::-1]
            filters = self.conv_transpose_filters
            
        for k in range(self._num_conv_layers):
            K = kernels[k]
            S = strides[k]
            P = 1
            out = []
            out.append(filters[k])
            for w in range(len(inp) -1):
                W = inp[w + 1]
                if encoder:
                    out.append(int(((W - K + (2 * P)) / S) + 1))
                else:
                    out.append(int((W - 1) * S - 2 * P  + K))
            inp = out
            if encoder:
                self.encoder_shape.append(out)
            else:
                self.decoder_shape.append(out)
    
    def _add_conv_transpose_layers(self):
        """Create all convolutional blocks in decoder."""
        for block_index in range(self._num_conv_layers):
            block_number = block_index + 1
            layer_index = 0
            self.layer2id['decoder']['conv_transpose{}'.format(block_number)] = 3 * block_index + layer_index 
            self.decoder.add_module(
                'conv_transpose{}'.format(block_number),
                self._add_conv_transpose_layer(block_index))
            layer_index += 1
            self.layer2id['decoder']['relu{}'.format(block_number)] = 3 * block_index + layer_index 
            self.decoder.add_module(
                'relu{}'.format(block_number),
                nn.ReLU())
            layer_index += 1
            self.layer2id['decoder']['batchnorm{}'.format(block_number)] = 3 * block_index + layer_index 
            self.decoder.add_module(
                'batchnorm{}'.format(block_number),
                nn.BatchNorm2d(self.conv_transpose_filters[block_index]))

    def _add_conv_transpose_layer(self, block_index):
        """Add a convolutional transpose block to a graph of layers, consisting of
        conv 2d + ReLU + batch normalization.
        """
        stride = self.conv_strides[::-1][block_index]
        kernel = self.conv_kernels[::-1][block_index]
        
        # if last transpose convolution layer, compute padding, output_padding and dilation
        # in order to match input dimension
        if block_index == self._num_conv_layers - 1:
            H_in = self.decoder_shape[block_index - 1][-2]
            W_in = self.decoder_shape[block_index - 1][-1]
            H_out = self.input_shape[-2]
            W_out = self.input_shape[-1]
            padding_0, output_padding_0, dilation_0 =  self._compute_padding(H_in, H_out, stride, kernel)
            padding_1, output_padding_1, dilation_1 =  self._compute_padding(W_in, W_out, stride, kernel)
            padding =(padding_0, padding_1)
            output_padding =(output_padding_0, output_padding_1)
            dilation =(dilation_0, dilation_1)
        else:
            padding = dilation = (1, 1)
            output_padding = (0, 0)
        
        conv_transpose_layer = nn.ConvTranspose2d(
            self.dec_input_channels[block_index],
            self.dec_output_channels[block_index],
            kernel,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            dilation=dilation
        )
        return conv_transpose_layer
    
    def _add_bottleneck(self):
        """Flatten encoder last conv layer output data
           and add bottleneck to latent dimension
        """
        self.shape_before_bottleneck = self.encoder_shape[self._num_conv_layers - 1]
        self.encoder.add_module('bottleneck_flatten', nn.Flatten(1))
        self.encoder.add_module('bottleneck_linear', 
                                nn.Linear(np.prod(self.shape_before_bottleneck), self.latent_space_dim * 2))
        self.encoder.add_module('bottleneck_batchnorm', nn.BatchNorm1d(self.latent_space_dim * 2))

    def _invert_bottleneck(self):
        """Reshape data from bottleneck latent dimension 
           to dimension of the output data after the encoder last conv layer
        """
        self.decoder.add_module('bottleneck_linear', 
                                nn.Linear(self.latent_space_dim, np.prod(self.shape_before_bottleneck)))
        self.decoder.add_module('bottleneck_batchnorm', nn.BatchNorm1d(np.prod(self.shape_before_bottleneck)))
        self.decoder.add_module('bottleneck_unflatten', nn.Unflatten(1, self.shape_before_bottleneck))
        
    def _compute_padding(self, dim_in, dim_out, stride, kernel):
        """"compute padding, output_padding and dilation in order to match
            input dimension in the last transpose convolution layer.
            dim_in is the second-last layer output shape
            dim_out is the first encoder layer input shape
        """
        stuff_list = []
    
        for subset in itertools.product(range(min(self.input_shape[-2:])), repeat=3):
            stuff_list.append(subset)
        
        for padding, output_padding, dilation in sorted(stuff_list, key=sum): 
            if (float((dim_in - 1) * stride - 2 * padding + 
                      dilation * (kernel - 1) + output_padding + 1).is_integer() and
                dilation > 0 and
                dim_out == (dim_in - 1) * stride - 2 * padding + dilation * (kernel - 1) + output_padding + 1):
                break   
        return padding, output_padding, dilation
                                
    def _get_encoder_layer(self, name):
        return self.encoder[self.layer2id['encoder'][name]]
    
    def _get_encoder_layers(self):
        return self.encoder
        
    def _get_decoder_layer(self, name):
        return self.decoder[self.layer2id['decoder'][name]]
        
    def _get_decoder_layers(self):
        return self.decoder
    
    def _get_output_shape(self, block_index, encoder=True):
        if encoder:
            return self.encoder_shape[block_index]
        else:
            return self.decoder_shape[block_index] 
    
    def reparameterise(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = std.new_empty(std.size()).normal_()
            return eps.mul_(std).add_(mu)
        else:
            return mu

    def forward(self, x):
        mu_logvar = self.encoder(x).view(-1, 2, self.latent_space_dim)
        mu = mu_logvar[:, 0, :]
        logvar = mu_logvar[:, 1, :]
        z = self.reparameterise(mu, logvar)
        return self.decoder(z), mu, logvar