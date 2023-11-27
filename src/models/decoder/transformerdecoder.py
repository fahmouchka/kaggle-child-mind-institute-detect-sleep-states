import torch
import torch.nn as nn
import torch.nn.functional as F

"""class TransformerDecoder(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        nhead: int,
        n_classes: int,
    ):
        super().__init__()
        self.conv = nn.Conv1d(input_size, hidden_size, 1)
        transformer_encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size, nhead=nhead, dropout=dropout, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            transformer_encoder_layer, num_layers=num_layers
        )
        self.linear = nn.Linear(hidden_size, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
    
        x = self.conv(x)  # (batch_size, n_channels, n_timesteps)
        x = x.transpose(1, 2)  # (batch_size, n_timesteps, n_channels)
        x = self.transformer_encoder(x)
        x = self.linear(x)  # (batch_size, n_timesteps, n_classes)

        return x"""

class TransformerDecoder(nn.Module):
    def __init__(
        self,
        input_size: int, 
        n_classes: int,
        num_layers: int,
        dropout: float , 
        nhead: int
    ):
        super(TransformerDecoder, self).__init__()
        #self.layer_normal = nn.LayerNorm(embed_dim)
        encoder_layers = [nn.TransformerEncoderLayer(input_size, nhead, input_size*4, dropout) for i in range(num_layers)]
        conv_layers = [nn.Conv1d(input_size,input_size,(num_layers-i)*2-1,stride=1,padding=0) for i in range(num_layers)]
        deconv_layers = [nn.ConvTranspose1d(input_size,input_size,(num_layers-i)*2-1,stride=1,padding=0) for i in range(num_layers)]
        layer_norm_layers = [nn.LayerNorm(input_size) for i in range(num_layers)]
        layer_norm_layers2 = [nn.LayerNorm(input_size) for i in range(num_layers)]
        self.transformer_encoder = nn.ModuleList(encoder_layers)
        self.conv_layers = nn.ModuleList(conv_layers)
        self.layer_norm_layers = nn.ModuleList(layer_norm_layers)
        self.layer_norm_layers2 = nn.ModuleList(layer_norm_layers2)
        self.deconv_layers = nn.ModuleList(deconv_layers)
        self.nhead = nhead
        self.pred = nn.Linear(input_size, n_classes)

    def forward(self, x):
        x = x.permute(2 ,0,1) 

        for conv, transformer_layer, layer_norm1, layer_norm2, deconv in zip(self.conv_layers,
                                                               self.transformer_encoder,
                                                               self.layer_norm_layers,
                                                               self.layer_norm_layers2,
                                                                self.deconv_layers):
            
            #LXBXC to BXCXL
            res=x
            x=F.relu(conv(x.permute(1,2,0)).permute(2,0,1))
            x=layer_norm1(x)
            x=transformer_layer(x)

            x=F.relu(deconv(x.permute(1,2,0)).permute(2,0,1))

            x=layer_norm2(x)
            x=res+x
            
            
        

        x = x.permute(1, 0, 2)


        output = self.pred(x)


        return output