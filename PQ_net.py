from quantize import *
# from swin_transformer import *
from collections import OrderedDict
from VQGAN_modules.model import Encoder, Decoder
from torch import nn
from loss.distortion import Distortion
from utils import *
from GAN.Discriminitor import Discriminator, Discriminator2
from channel import convention_channel,BSC_Channel
class PQNet(nn.Module):
    def __init__(
        self,
        config
    ):
        super().__init__()
        # embed_dim = config.embed_dim
        self.config = config
        embed_dim = config.ga_kwargs['embed_dims'][-1]

        n_embed = config.embed_n
        chunk_num = config.chunk_num
        self.ga = Encoder(double_z= False, z_channels= 256, resolution=256,in_channels= 3,
                        out_ch=3, ch= 128, ch_mult=[1,1,2,2,4], num_res_blocks=2,
                          attn_resolutions=[16] ,dropout=0.0)
        # self.ga = Encoder(double_z= False, z_channels= 256, resolution=256,in_channels= 3,
        #                 out_ch=3, ch= 128, ch_mult=[1,2,2,4], num_res_blocks=2,
        #                   attn_resolutions=[16] ,dropout=0.0)
        self.gs = Decoder(double_z= False, z_channels= 256, resolution=256,in_channels= 3,
                        out_ch=3, ch= 128, ch_mult=[1,1,2,2,4], num_res_blocks=2,
                          attn_resolutions=[16] ,dropout=0.0)
        # self.gs = Decoder(double_z= False, z_channels= 256, resolution=256,in_channels= 3,
        #                 out_ch=3, ch= 128, ch_mult=[1,2,2,4], num_res_blocks=2,
        #                   attn_resolutions=[16] ,dropout=0.0)
        self.quant_conv = torch.nn.Conv2d(256, embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, 256, 1)

        self.quantize = PQuantize(embed_dim, n_embed, chunk_num, share_codebook=config.share_codebook, config=config)
        self.H = self.W = 0

        self.latent_loss_weight=self.config.latent_loss_weight
        self.distortion=Distortion(config)
        self.disc_factor = self.config.disc_factor

        self.discriminator = Discriminator2(config)
        self.BSC_channel = BSC_Channel(config)


    def forward(self, input, *args, **kwargs):
        B, C, H, W = input.shape
        # self.update_resolution(H, W)

        quant, diff_t , id, id_prob, y = self.encode(input)
        if self.training != True:
            B,  H, W, C = id.shape
            id_hat = self.BSC_channel.BSC_bit_flip(id.view(B, -1), ber=0.0)
            id_hat = id_hat.reshape(B, H, W,C)
            # id_hat[:,:,:,1:] = 0

        if self.training != True:
            dec_noise = self.decode_code(id_hat)
            dec_wo_noise = self.decode_code(id)
        else:
            dec_noise = self.decode(quant)
            dec_wo_noise = dec_noise



        # ntc_loss = self.distortion(dec_wo_noise, input)
        mse_loss = self.distortion(dec_noise,input) / (255)
        lp_loss = LPIPS_vgg(dec_noise, input, normalize = True).mean()

        recon_loss=(0.1 * mse_loss +  lp_loss)
        # recon_loss = 0.1* mse_loss
        # recon_loss = mse_loss
        # recon_loss = lp_loss

        disc_fake = self.discriminator(dec_noise)
        disc_real = self.discriminator(input)

        g_loss = -torch.mean(disc_fake)

        if self.training == True and self.config.use_gan == True:
            lam = self.calculate_lambda(lp_loss, g_loss)
            disc_factor = self.adopt_weight(self.disc_factor, kwargs['epoch'], 0, 0.)
        else:
            lam = 0
            disc_factor = 0
        bpp = 8 * self.config.chunk_num / 256
        loss=  recon_loss + self.latent_loss_weight*diff_t + disc_factor * lam *  g_loss

        d_loss_real = torch.mean(F.relu(1. - disc_real))
        d_loss_fake = torch.mean(F.relu(1. + disc_fake))
        gan_loss =  0.5*(d_loss_real + d_loss_fake)
        # gan_loss = 0

        return_dict={'dec':dec_noise,'loss': loss, 'pure_dec':dec_wo_noise, 'bpp':bpp, 'gan_loss': gan_loss}
        # get_heat_map(indexes.reshape(H // 16, W // 16).cpu().numpy(), save_folder = 'heatmap')
        return return_dict

    def encode(self, input):
        input = 2 * input - 1
        y = self.ga(input)
        y = self.quant_conv(y)
        quant, diff, id, id_prob = self.quantize(y)
        diff_t = diff.unsqueeze(0)

        return quant, diff_t , id, id_prob, y


    def decode(self, quant):
        quant = self.post_quant_conv(quant)
        dec = self.gs(quant)
        dec = (dec+1) / 2
        return dec.clamp(0,1)
    def decode_code(self, id):
        quant = self.quantize.de_quantize_code(id)
        quant = self.post_quant_conv(quant)
        dec = self.gs(quant)
        dec = (dec+1) / 2
        return dec.clamp(0,1)

    def load_checkpoint(self, config):
        path = config.checkpoint
        state_dict = torch.load(path)
        result_dict = {}
        for key, weight in state_dict.items():
            if 'VQVAE' in key:
                result_key = key.replace('VQVAE.', '')
            else:
                result_key = key
            if 'attn_mask' not in key and 'rate_adaption.mask' not in key:
                result_dict[result_key] = weight
        print(self.load_state_dict(result_dict, strict=False))
        del result_dict, state_dict

    def init_pre_trained_model(self, ckpt_path):
        para = torch.load(ckpt_path, weights_only=False)
        state_dict = para['state_dict']
        # state_dict = para['model']
        encoder_state_dict = OrderedDict()
        decoder_state_dict = OrderedDict()
        quant_conv_dict = OrderedDict()
        post_conv_dict = OrderedDict()

        for k, v in state_dict.items():
            if k.startswith('first_stage_model.encoder.'):
                encoder_state_dict[k[len('first_stage_model.encoder.'):]] = v
            elif k.startswith('first_stage_model.decoder.'):
                decoder_state_dict[k[len('first_stage_model.decoder.'):]] = v
            elif k.startswith('first_stage_model.quant_conv.'):
                quant_conv_dict[k[len('first_stage_model.quant_conv.'):]] = v
            elif k.startswith('first_stage_model.post_quant_conv.'):
                post_conv_dict[k[len('first_stage_model.post_quant_conv.'):]] = v
        self.ga.load_state_dict(encoder_state_dict)
        self.gs.load_state_dict(decoder_state_dict)
        self.quant_conv.load_state_dict(quant_conv_dict)
        self.post_quant_conv.load_state_dict(post_conv_dict)
    def init_pre_trained_model_imagenet(self, ckpt_path):
        para = torch.load(ckpt_path, weights_only=False)
        # state_dict = para['state_dict']
        state_dict = para['model']
        encoder_state_dict = OrderedDict()
        decoder_state_dict = OrderedDict()
        quant_conv_dict = OrderedDict()
        post_conv_dict = OrderedDict()

        for k, v in state_dict.items():
            if k.startswith('content_codec.enc.encoder.'):
                encoder_state_dict[k[len('content_codec.enc.encoder.'):]] = v
            elif k.startswith('content_codec.dec.decoder.'):
                decoder_state_dict[k[len('content_codec.dec.decoder.'):]] = v
            elif k.startswith('content_codec.enc.quant_conv.'):
                quant_conv_dict[k[len('content_codec.enc.quant_conv.'):]] = v
            elif k.startswith('content_codec.dec.post_quant_conv.'):
                post_conv_dict[k[len('content_codec.dec.post_quant_conv.'):]] = v
        self.ga.load_state_dict(encoder_state_dict)
        self.gs.load_state_dict(decoder_state_dict)
        self.quant_conv.load_state_dict(quant_conv_dict)
        self.post_quant_conv.load_state_dict(post_conv_dict)
    def configure_optimizers(self, config):
        lr = config.lr_codec
        opt_vq = torch.optim.Adam(
            list(self.ga.parameters()) +
            list(self.gs.parameters()) +
            list(self.quant_conv.parameters()) +
            list(self.post_quant_conv.parameters()) +
            list(self.quantize.parameters()),
            lr=lr
        )
        opt_disc = torch.optim.Adam(self.discriminator.parameters(),
                                    lr=lr)
        return opt_vq, opt_disc

    def calculate_lambda(self, perceptual_loss, gan_loss):
        last_layer = self.gs.conv_out
        last_layer_weight = last_layer.weight
        perceptual_loss_grads = torch.autograd.grad(perceptual_loss, last_layer_weight, retain_graph=True)[0]
        gan_loss_grads = torch.autograd.grad(gan_loss, last_layer_weight, retain_graph=True)[0]
        λ = torch.norm(perceptual_loss_grads) / (torch.norm(gan_loss_grads) + 1e-4)
        λ = torch.clamp(λ, 0, 1e4).detach()
        return 0.8 * λ

    @staticmethod
    def adopt_weight(disc_factor, i, threshold, value=0.):
        if i < threshold:
            disc_factor = value
        return disc_factor
