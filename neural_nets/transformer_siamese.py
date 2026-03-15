import torch.nn as nn
import torch.nn.functional as F
import torch
import pdb
from .unet_base_blocks import Conv1x3x1
from .utils import pad_to_power_of_2
from einops import rearrange
## Siamese Unet with Learnable Channel attention

# Importance Weighted Channel Attention
class IWCA(nn.Module):
    def __init__(self, in_channels):
        super(IWCA, self).__init__()
        # no mixing of channel information
        self.c0 = nn.Conv2d(in_channels, in_channels, 
                                    kernel_size=3, groups=in_channels, padding=1)
        self.bn0 = nn.BatchNorm2d(in_channels)
        self.c1 = nn.Conv2d(in_channels, in_channels, 
                                    kernel_size=1, groups=in_channels)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.sigmoid = nn.Sigmoid()
        self.importance_wts = None

    def forward(self, x):
        # Group Convolution
        x_c = self.bn0(F.relu(self.c0(x)))
        x_c = self.bn1(F.relu(self.c1(x_c)))
        # Global Average Pooling
        x_avg = self.global_avg_pool(x_c)
        importance_weights = self.sigmoid(x_avg)
        self.importance_wts = importance_weights
        # Scale the original input
        out = x * importance_weights
        return out


class Down1x3x1(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.l1 = Conv1x3x1(in_channels, 64)
        self.l2 = Conv1x3x1(64, 128)
        self.l3 = Conv1x3x1(128, out_channels)
        
    def forward(self, x):
        x1 = self.l1(x)
        x2 = self.l2(x1)
        x3 = self.l3(x2)
        return x3, [x1, x2]
    
class UpConcat(nn.Module):
    def __init__(self, in_channels) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels*2, in_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(in_channels)
        
    def forward(self, msi_feat, hsi_feat):
        # upsample msi features
        sx, sy = msi_feat.shape[-2] // hsi_feat.shape[-2], msi_feat.shape[-1] // hsi_feat.shape[-1]
        hsi_feat = F.interpolate(hsi_feat, scale_factor=(sx, sy))
        out = torch.cat([hsi_feat, msi_feat], dim=1)
        return self.bn(F.relu(self.conv(out)))
        
        
        
class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # upsample latent to match spatial extent
        # self.conv = nn.Conv2d(in_channels*2, in_channels, kernel_size=3, padding=1)
        # self.bn = nn.BatchNorm2d(in_channels)
        self.deconv3 = nn.ConvTranspose2d(in_channels, 128, 
                                          kernel_size=3, 
                                          stride=2, padding=1, output_padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.deconv2 = nn.ConvTranspose2d(128*2, 64, kernel_size=3, 
                                          stride=2, padding=1, output_padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.deconv1 = nn.ConvTranspose2d(64*2, out_channels, 
                                          kernel_size=3, 
                                          stride=2, padding=1, output_padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)

    def forward(self, x, skip_connection):
        
        # x = self.bn(F.relu(self.conv(z)))
        x = self.bn3(F.relu(self.deconv3(x)))
        x = torch.cat((x, skip_connection[1]), dim=1) 
        x = self.bn2(F.relu(self.deconv2(x)))
        x = torch.cat((x, skip_connection[0]), dim=1)
        x = self.bn1(F.relu(self.deconv1(x)))
        return x
    
            
class Up1x3x1(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # upsample latent to match spatial extent
        self.conv = nn.Conv2d(in_channels*2, in_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(in_channels)
        # up 1x3x1
        self.deconv3 = nn.ConvTranspose2d(in_channels, 128, 
                                          kernel_size=1, stride=2, output_padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.deconv2 = nn.ConvTranspose2d(128*2, 64, kernel_size=3, 
                                          stride=2, padding=1, output_padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.deconv1 = nn.ConvTranspose2d(64*2, out_channels, 
                                          kernel_size=1, stride=2, output_padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)

    def forward(self, z, skip_connection):
        x = self.bn(F.relu(self.conv(z)))
        x = self.bn3(F.relu(self.deconv3(x)))
        x = torch.cat((x, skip_connection[1]), dim=1) 
        x = self.bn2(F.relu(self.deconv2(x)))
        x = torch.cat((x, skip_connection[0]), dim=1)
        x = self.bn1(F.relu(self.deconv1(x)))
        return x


class SiameseEncoder(nn.Module):
    def __init__(self, hsi_in, msi_in, latent_dim):
        super().__init__()
        # hsi_enc -> 31 x h x w -> 256  x 1 x 1
        self.channel_selector = IWCA(hsi_in)
        self.hsi_enc = Down1x3x1(hsi_in, latent_dim)
        # msi_enc -> 3 x H x W -> -> 256  x 1 x 1
        self.msi_enc = Down1x3x1(msi_in, latent_dim)
        
    def forward(self, hsi, msi):
        hsi = self.channel_selector(hsi)
        z_hsi, hsi_out = self.hsi_enc(hsi)
        z_msi, msi_out = self.msi_enc(msi) # apply bilinear upsample here
        # get scale of upsampling
        sx, sy = z_msi.shape[-2] // z_hsi.shape[-2], z_msi.shape[-1] // z_hsi.shape[-1]
        z_hsi = F.interpolate(z_hsi, scale_factor=(sx, sy))
        return z_hsi, z_msi, hsi_out, msi_out


class SegmentationDecoder(nn.Module):
    def __init__(self, latent_dim, out_channels):
        super().__init__()
        self.upcat2 = UpConcat(latent_dim//2)# [B, 128, 64, 64]
        self.upcat1 = UpConcat(latent_dim//4)# [B, 64, 128, 128]
        # self.upcat2 = CrossAttentionBlock(
        #     hsi_channels=latent_dim//2, 
        #     msi_channels=latent_dim//2, out_channels=latent_dim//2)
        # self.upcat1 = CrossAttentionBlock(
        #     hsi_channels=latent_dim//4, 
        #     msi_channels=latent_dim//4, out_channels=latent_dim//4)
        self.decoder = Up(latent_dim, out_channels)

    def forward(self, z, hsi_out, msi_out):
        # merge outputs of hsi and msi encoder
        out2 = self.upcat2(msi_out[1], hsi_out[1])
        out1 = self.upcat1(msi_out[0], hsi_out[0])
        x = self.decoder(z, [out1, out2])
        return x


class AttentionBlock(nn.Module):
    def __init__(self, in_channels):
        super(AttentionBlock, self).__init__()
        self.query = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.key = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.value = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch_size, C, height, width = x.size()
        proj_query = self.query(x).view(batch_size, -1, width * height).permute(0, 2, 1)
        proj_key = self.key(x).view(batch_size, -1, width * height)
        energy = torch.bmm(proj_query, proj_key)
        attention = F.softmax(energy, dim=-1)
        proj_value = self.value(x).view(batch_size, -1, width * height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(batch_size, C, height, width)
        out = self.gamma * out + x
        return out

def fourier_transform(x):
    return torch.fft.fft2(x)

def inverse_fourier_transform(x):
    return torch.fft.ifft2(x)


class ReduceFourierDimLinear(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ReduceFourierDimLinear, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        # x: [B, P, C, F]
        B, P, C, F = x.shape
        x = x.view(B * P * C, F)  # Flatten dimensions except the last one
        x = self.linear(x)
        x = x.view(B, P, C, -1)  # Reshape back to [B, P, C, reduced_dim]
        return x

class FourierCrossAttention(nn.Module):
    def __init__(self, input_dim):
        super(FourierCrossAttention, self).__init__()
        self.input_dim = input_dim
        self.query_real = nn.Linear(input_dim, input_dim)
        self.query_imag = nn.Linear(input_dim, input_dim)
        self.key_real = nn.Linear(input_dim, input_dim)
        self.key_imag = nn.Linear(input_dim, input_dim)
        self.value_real = nn.Linear(input_dim, input_dim)
        self.value_imag = nn.Linear(input_dim, input_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, y):
        # Apply Fourier Transform
        x_freq = fourier_transform(x)
        y_freq = fourier_transform(y)
        
        # Split into real and imaginary parts
        x_real, x_imag = x_freq.real, x_freq.imag
        y_real, y_imag = y_freq.real, y_freq.imag
        
        # Ensure the dimensions are compatible
        x_real = x_real.view(-1, self.input_dim)
        x_imag = x_imag.view(-1, self.input_dim)
        y_real = y_real.view(-1, self.input_dim)
        y_imag = y_imag.view(-1, self.input_dim)
        
        # Compute queries, keys, and values for real and imaginary parts
        Q_real = self.query_real(x_real) - self.query_imag(x_imag)
        Q_imag = self.query_real(x_imag) + self.query_imag(x_real)
        
        K_real = self.key_real(y_real) - self.key_imag(y_imag)
        K_imag = self.key_real(y_imag) + self.key_imag(y_real)
        
        V_real = self.value_real(y_real) - self.value_imag(y_imag)
        V_imag = self.value_real(y_imag) + self.value_imag(y_real)
        
        # Reshape back to batch form for attention computation
        b, n = x.shape[0], x.shape[2] * x.shape[3]
        Q_real = Q_real.view(b, n, -1)
        Q_imag = Q_imag.view(b, n, -1)
        K_real = K_real.view(b, n, -1)
        K_imag = K_imag.view(b, n, -1)
        V_real = V_real.view(b, n, -1)
        V_imag = V_imag.view(b, n, -1)
        
        # Compute attention
        attention_scores_real = torch.matmul(Q_real, K_real.transpose(-2, -1)) - torch.matmul(Q_imag, K_imag.transpose(-2, -1))
        attention_scores_imag = torch.matmul(Q_real, K_imag.transpose(-2, -1)) + torch.matmul(Q_imag, K_real.transpose(-2, -1))
        attention_scores = torch.sqrt(attention_scores_real**2 + attention_scores_imag**2)
        
        attention_weights = self.softmax(attention_scores)
        
        attention_output_real = torch.matmul(attention_weights, V_real)
        attention_output_imag = torch.matmul(attention_weights, V_imag)
        
        # Combine real and imaginary parts
        attention_output = attention_output_real + 1j * attention_output_imag
        
        # Inverse Fourier Transform
        attention_output = inverse_fourier_transform(attention_output)
        return torch.view_as_real(attention_output)




    
class CustomTransformerDecoderWithFourier(nn.Module):
    def __init__(self, input_dim, num_classes, patch_size, input_size):
        super(CustomTransformerDecoderWithFourier, self).__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.patch_size = patch_size
        self.input_size = input_size

        # Cross Attention with Fourier
        self.cross_attention = FourierCrossAttention(input_dim)
        
        # Transformer decoder configuration
        decoder_layer = nn.TransformerDecoderLayer(d_model=input_dim, nhead=8)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)
        self.linear = nn.Linear(input_dim, num_classes)

    def forward(self, encoder_output, original_shape):
        # Reshape encoder output to sequence format
        b, c, h, w = encoder_output.size()
        n = h * w
        src = encoder_output.view(b, c, n).permute(2, 0, 1)  # (n, b, c)
        # Create a sequence of positional encodings matching the original spatial dimensions
        tgt = torch.zeros((n, b, c), device=encoder_output.device)
        
        # Apply Cross Attention in Fourier domain
        src = self.cross_attention(encoder_output, encoder_output)
        # [2048, 4, 512, 2]
        # Pass through transformer decoder
        decoder_output = self.transformer_decoder(tgt, src)
        # Reshape back to image format
        decoder_output = decoder_output.permute(1, 0, 2).contiguous()
        decoder_output = decoder_output.view(b, h, w, -1)
        decoder_output = decoder_output.permute(0, 3, 1, 2)  # (b, d, h, w)
        # Apply final linear layer to match number of classes
        output = self.linear(decoder_output)
        return output
    
class CrossAttentionBlock(nn.Module):
    def __init__(self, hsi_channels, msi_channels, out_channels):
        super(CrossAttentionBlock, self).__init__()
        self.query = nn.Conv2d(hsi_channels, out_channels, kernel_size=1)
        self.key = nn.Conv2d(msi_channels, out_channels, kernel_size=1)
        self.value = nn.Conv2d(msi_channels, out_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, z_hsi, z_msi):
        if z_hsi.shape[-2] > z_msi.shape[-2]:
            sx, sy = z_hsi.shape[-2] // z_msi.shape[-2], z_hsi.shape[-1] // z_msi.shape[-1]
            z_msi = F.interpolate(z_msi, scale_factor=(sx, sy))
        
        batch_size, _, height, width = z_hsi.size()
        
        # Project HSI embeddings to queries
        proj_query = self.query(z_hsi).view(batch_size, -1, height * width).permute(0, 2, 1)
        
        # Project MSI embeddings to keys and values
        proj_key = self.key(z_msi).view(batch_size, -1, height * width)
        proj_value = self.value(z_msi).view(batch_size, -1, height * width)

        # Compute attention map
        energy = torch.bmm(proj_query, proj_key)
        attention = F.softmax(energy, dim=-1)

        # Apply attention map to values
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(batch_size, -1, height, width)

        # Combine with the original HSI embedding using the gamma parameter
        out = self.gamma * out + z_hsi
        
        return out

class CASiameseTransformer(nn.Module):
    def __init__(self, hsi_in, msi_in, latent_dim, output_channels, **kwargs):
        super().__init__()
        self.encoder = SiameseEncoder(hsi_in, msi_in, latent_dim)
        self.attention = AttentionBlock(latent_dim)
        self.cross_attention = CrossAttentionBlock(
            hsi_channels=latent_dim, msi_channels=latent_dim, out_channels=latent_dim)
        self.decoder = CustomTransformerDecoderWithFourier(latent_dim * 2,
                                                           num_classes=output_channels,
                                                           patch_size=16,
                                                           input_size=16)
        
    def forward(self, hsi, msi):
        orig_ht, orig_width = msi.shape[2:]
        hsi = hsi.to(torch.double)
        msi = msi.to(torch.double)
        msi = pad_to_power_of_2(msi)
        hsi = pad_to_power_of_2(hsi)
        
        z_hsi, z_msi, hsi_out, msi_out = self.encoder(hsi, msi)
        z = torch.cat([z_hsi, z_msi], dim=1)
        # Apply attention
        # z_hsi = self.attention(z_hsi)
        # z_msi = self.attention(z_msi)
        # z = self.cross_attention(z_hsi, z_msi)
        segmentation_map = self.decoder(z, (orig_ht, orig_width))  
        outputs = {
            'preds': segmentation_map[:, :, :orig_ht, :orig_width],
            'embeddings': [z_hsi, z_msi]
        }  
        return outputs


if __name__ == '__main__':
    # usage
    model = CASiameseTransformer(31, 3, 256, 5)  # Assume output channels for segmentation map is 5
    for i in range(1, 5):
        hsi = torch.rand(2, 31, 64*i, 64*i)
        msi = torch.rand(2, 3, 256*i, 256*i)
        output = model(hsi, msi)
        print(output.shape)
        # instead of output, we will use the loss to compute which channel 
        # inflences the training more than others
        # lower loss also means that those channels are better
        # full jacobian -> [2, 5, 256, 256, 2, 31, 1, 1] so take mean 
        # jacobian computation
        # output = model(hsi, msi)
        # jacobian = torch.autograd.functional.jacobian(lambda x: output.mean((2, 3)), 
        #                                 model.encoder.channel_selector.importance_wts)
        # jacobian = jacobian.squeeze() # [2, 5, 2, 31]
        # jacobian = jacobian.mean((0, 2)) # [5, 31]
        # print(jacobian.shape)
        

