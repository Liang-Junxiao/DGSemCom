import torch
from torch import nn
import torch.nn.functional as F
from PQ_net import PQNet
from diffusion_modules.transformer import Transformer2DModel
from diffusion_modules.vq_scheduler import VQDiffusionScheduler
from diffusion_modules.utils import *
from channel.channel import convention_channel
import time
import tqdm
class DGSemCom(nn.Module):
    def __init__(self, config):
        super(DGSemCom, self).__init__()
        self.config = config
        self.device = config.device
        self.PQNet = PQNet(config).to(self.device)
        self.transformer = Transformer2DModel(config).to(self.device)
        self.schedule = VQDiffusionScheduler(config,
                                             num_vec_classes= config.embed_n +1,
                                             num_train_timesteps= config.num_timesteps).to(self.device)
        self.channel_trans =BSC_Channel(config).to(self.device)
        self.convention_channel = convention_channel(config)
        self.alpha_t = None
        self.latent_size = config.latent_size
        self.embed_n = config.embed_n
        self.num_timesteps = config.num_timesteps
        self.num_classes = config.embed_n + 1

        self.receive_error_num = 0
        self.receive_error_rate= 0
        self.correction_error_num = 0
        self.correction_error_rate = 0
        self.infer_num = 0
        self.register_buffer('Lt_history', torch.zeros(self.num_timesteps))
        self.register_buffer('Lt_count', torch.zeros(self.num_timesteps))
        for i in range(self.config.chunk_num):
            self.register_buffer('col_ind'+str(i), None)


    def forward(self, input_img):
        # get the discrete latent representation y_latent and rate allocation indexes
        B, C, H, W = input_img.shape
        with torch.no_grad():
            _, _ , x_0, _, _ = self.PQNet.encode(input_img)
        H, W, N = self.latent_size[0], self.latent_size[1] , self.latent_size[2]

        x_0 = x_0.reshape(B, H * W, N)

        kl_loss, aux_loss = self._train_loss(x_0)

        kl_loss = kl_loss.sum()
        aux_loss = aux_loss.sum()
        total_loss = aux_loss + kl_loss
        total_loss = total_loss / (B * H * W * N)
        return_dict = {
            'total_loss': total_loss,
            'lb_loss': kl_loss,
            'aux_loss': aux_loss
        }
        return return_dict

    def _train_loss(self, x ):  # get the KL loss
        b, device = x.size(0), x.device

        x_start = x
        t, pt = self.sample_time(b, device, 'importance')

        x_t = self.schedule.sample_with_t(x_start, time_step =t)

        ############### go to p_theta function ###############
        log_p_x_0_predict = self.transformer(x_t, timestep= t)

        kl_loss, aux_loss = self.schedule(log_p_x_0_predict, x_start, x_t, timestep = t)

        Lt2 = kl_loss.pow(2)
        Lt2_prev = self.Lt_history.gather(dim=0, index=t)
        new_Lt_history = (0.1 * Lt2 + 0.9 * Lt2_prev).detach()
        self.Lt_history.scatter_(dim=0, index=t, src=new_Lt_history)
        self.Lt_count.scatter_add_(dim=0, index=t, src=torch.ones_like(Lt2))

        kl_loss = kl_loss / pt
        aux_loss = aux_loss / pt

        return kl_loss, aux_loss
    def sample_time(self, b, device, method='uniform'):
        if method == 'importance':
            if not (self.Lt_count > 10).all():
                return self.sample_time(b, device, method='uniform')
            Lt_sqrt = torch.sqrt(self.Lt_history + 1e-10) + 0.0001
            Lt_sqrt[0] = Lt_sqrt[1]  # Overwrite decoder term with L1.
            pt_all = Lt_sqrt / Lt_sqrt.sum()
            t = torch.multinomial(pt_all, num_samples=b, replacement=True)
            pt = pt_all.gather(dim=0, index=t)
            return t, pt
        elif method == 'uniform':
            t = torch.randint(0, self.num_timesteps, (b,), device=device).long()

            pt = torch.ones_like(t).float() / self.num_timesteps
            return t, pt
        else:
            raise ValueError

    def load_vq_model(self, path):
        pretrained = torch.load(path)
        result_dict = {}
        for key, weight in pretrained.items():
            result_key = key
            result_dict[result_key] = weight
        print(self.PQNet.load_state_dict(result_dict, strict=False))
        del result_dict, pretrained


    def posterior_sample(self, num_inference_steps: int = 100,
        received_signal: torch.Tensor = None,
        latents: torch.Tensor = None,
        indexes = None,
        truncation_rate: float = 1.0,
        prob = None
        ):
        if received_signal is None:
            do_uncond_gen = True
            batch_size = 1
        else:
            do_uncond_gen = False
            batch_size = received_signal.shape[0]

        if prob is None:
            use_hard = True
            use_soft = False
        else:
            use_hard = False
            use_soft = True
        # get the initial completely masked latents unless the user supplied it
        H, W, N = self.latent_size[0], self.latent_size[1] , self.latent_size[2]

        latents_shape = (batch_size, H * W, N)

        mask_class = self.embed_n

        if latents is None:
            latents = torch.full(latents_shape, mask_class).to(self.device)

        self.schedule.set_timesteps(num_inference_steps, device=self.device)
        timesteps_tensor = self.schedule.timesteps.to(self.device)


        sample = latents

        for i, t in enumerate(timesteps_tensor):
            latent_model_input = sample
            t = t.reshape(batch_size, )
            model_output = self.transformer(latent_model_input, timestep=t)

            if not do_uncond_gen:
                model_output = model_output.reshape(batch_size, H * W * N, -1)
                model_output = model_output.permute(0,2,1)  # [B, N, L]  N = 1024, L= seq_len
                model_output_prob = torch.exp(model_output)


                gamma =1
                if t>0:
                    if use_hard:
                        p_y_given_x0 = self.channel_trans.transition_matrix[received_signal].permute(0, 2, 1)
                    else:
                        p_y_given_x0 = prob.unsqueeze(0).permute(0,2,1).to(self.device)
                    p_x0_given_xt_y = p_y_given_x0** gamma * model_output_prob

                    #p_y_given_xt = torch.gather(transition_output, dim=1, index=received_signal.unsqueeze(0))
                    # p_x0_given_xt_y = p_x0_given_xt_y / p_y_given_xt   # normalize， use F.normalize
                    p_x0_given_xt_y = F.normalize(p_x0_given_xt_y, dim = 1)

                    model_output = torch.log(p_x0_given_xt_y)
                else:
                    model_output = model_output

                # model_output = self.vq_diff.truncate(model_output, truncation_rate)
                # remove `log(0)`'s (`-inf`s)
                # model_output = model_output.clamp(-70)
                # compute the previous noisy sample x_t -> x_t-1
                model_output = model_output.permute(0, 2, 1)
                model_output = model_output.reshape(batch_size, H*W, N, -1 )

                model_output = self.truncate(model_output, truncation_rate)
                # remove `log(0)`'s (`-inf`s)
                model_output = model_output.clamp(-70)

                sample = self.schedule.step(model_output, timestep=t, sample=sample,
                                                     generator=None)

                # diff = torch.count_nonzero(sample.reshape(batch_size,-1) - received_signal).item()
                # print(diff)

            if do_uncond_gen:
                model_output = self.truncate(model_output, truncation_rate)
                # remove `log(0)`'s (`-inf`s)
                model_output = model_output.clamp(-70)
                # compute the previous noisy sample x_t -> x_t-1
                sample = self.schedule.step(model_output, timestep=t, sample=sample, generator= None)


        sample = sample.reshape(batch_size , H, W, N)

        image = self.PQNet.decode_code(sample)
        return image

    def posterior_sample_star(self, num_inference_steps: int = 100,
        received_signal: torch.Tensor = None,
        latents: torch.Tensor = None,
        indexes = None,
        truncation_rate: float = 1.0,
        prob = None,
        delta_t = 1,
        ori_signal = None
        ):

        correct_start_time = time.time()

        if received_signal is None:
            do_uncond_gen = True
            batch_size = 1
        else:
            do_uncond_gen = False
            batch_size = received_signal.shape[0]

        if prob is None:
            use_hard = True
            use_soft = False
        else:
            use_hard = False
            use_soft = True
        # get the initial completely masked latents unless the user supplied it
        H, W, N = self.latent_size[0], self.latent_size[1] , self.latent_size[2]

        latents_shape = (batch_size, H * W, N)

        mask_class = self.embed_n

        if latents is None:
            latents = torch.full(latents_shape, mask_class).to(self.device)

        self.schedule.set_timesteps(num_inference_steps, device=self.device)
        timesteps_tensor = self.schedule.timesteps.to(self.device)


        sample = latents

        # for i, t in enumerate(timesteps_tensor):
        t= torch.tensor(99, device=self.device, dtype=torch.int32)
        stop = False
        while stop == False:
            latent_model_input = sample
            t = t.reshape(batch_size, )
            model_output = self.transformer(latent_model_input, timestep=t)  #p（z0|zt）

            if not do_uncond_gen:
                model_output = model_output.reshape(batch_size, H * W * N, -1)
                model_output = model_output.permute(0,2,1)  # [B, N, L]  N = 1024, L= seq_len
                model_output_prob = torch.exp(model_output)

                # if use_hard:
                #     transition_output = torch.matmul(self.channel_trans.transition_matrix, model_output_prob)
                # elif use_soft:
                #     transition_output = torch.matmul()    # for normalize, use F.normalize

                gamma =1
                if t>0:
                    if use_hard:
                        p_y_given_x0 = self.channel_trans.transition_matrix[received_signal].permute(0, 2, 1)
                    else:
                        p_y_given_x0 = prob.unsqueeze(0).permute(0,2,1).to(self.device)

                    p_x0_given_xt_y = p_y_given_x0** gamma * model_output_prob

                    p_x0_given_xt_y = F.normalize(p_x0_given_xt_y, dim = 1, p=1)

                    model_output = torch.log(p_x0_given_xt_y)
                else:
                    model_output = model_output

                # model_output = self.vq_diff.truncate(model_output, truncation_rate)
                # remove `log(0)`'s (`-inf`s)
                # model_output = model_output.clamp(-70)
                # compute the previous noisy sample x_t -> x_t-1
                model_output = model_output.permute(0, 2, 1)
                model_output = model_output.reshape(batch_size, H*W, N, -1 )

                model_output = self.truncate(model_output, truncation_rate)
                # remove `log(0)`'s (`-inf`s)
                model_output = model_output.clamp(-70)

                sample = self.schedule.step_star(model_output, timestep=t, sample=sample,
                                                     generator=None)

                # diff = torch.count_nonzero(sample.reshape(batch_size,-1) - received_signal).item()
                # print(diff)

            if do_uncond_gen:
                model_output = self.truncate(model_output, truncation_rate)
                # remove `log(0)`'s (`-inf`s)
                model_output = model_output.clamp(-70)
                # compute the previous noisy sample x_t -> x_t-1
                sample = self.schedule.step_star(model_output, timestep=t, sample=sample, generator= None, delta_t = delta_t)


            if t==0:
                stop = True
            t = (t- delta_t).clamp(min=0)




        sample = sample.reshape(batch_size , H, W, N)


        image = self.PQNet.decode_code(sample)

        error_num = (received_signal - ori_signal).ne(0).float().sum()
        correction_num = (sample.view(batch_size,-1) - ori_signal).ne(0).float().sum()
        self.receive_error_num += error_num
        self.correction_error_num += correction_num
        self.infer_num += 1
        self.receive_error_rate = self.receive_error_num / (self.infer_num*batch_size*H*W*N)
        self.correction_error_rate = self.correction_error_num / (self.infer_num*batch_size*H*W*N)

        return image


    def truncate(self, log_p_x_0: torch.FloatTensor, truncation_rate: float) -> torch.FloatTensor:
        """
        Truncates log_p_x_0 such that for each column vector, the total cumulative probability is `truncation_rate` The
        lowest probabilities that would increase the cumulative probability above `truncation_rate` are set to zero.
        """
        B, L, C, N = log_p_x_0.shape
        log_p_x_0 = log_p_x_0.reshape(B, L*C, N ).permute(0,2,1)
        sorted_log_p_x_0, indices = torch.sort(log_p_x_0, 1, descending=True)
        sorted_p_x_0 = torch.exp(sorted_log_p_x_0)
        keep_mask = sorted_p_x_0.cumsum(dim=1) < truncation_rate

        # Ensure that at least the largest probability is not zeroed out
        all_true = torch.full_like(keep_mask[:, 0:1, :], True)
        keep_mask = torch.cat((all_true, keep_mask), dim=1)
        keep_mask = keep_mask[:, :-1, :]

        keep_mask = keep_mask.gather(1, indices.argsort(1))

        rv = log_p_x_0.clone()

        rv[~keep_mask] = -torch.inf  # -inf = log(0)
        rv = rv.permute(0,2,1)
        rv = rv.reshape(B, L, C, N)
        return rv



class BSC_Channel(nn.Module):
    def __init__(self, config):
        super(BSC_Channel, self).__init__()
        bit_num = config.bit_num
        self.device = config.device
        self.bit_num = bit_num
        self.symbols = 2 ** bit_num
        self.transition_matrix = None
        self.ber = None

        self.weights = torch.tensor([[2**i] for i in range(config.bit_num)], dtype=torch.float32, device=config.device)
    def calculate_transition_matrix(self, ber):
        if self.transition_matrix is not None  and ber == self.ber:
            return self.transition_matrix, self.ber
        symbols = torch.arange(self.symbols)
        symbol_bits = symbols.unsqueeze(1).bitwise_and(1 << torch.arange(self.bit_num)).ne(0).long()

        # error_probs = torch.tensor([self.comb(self.bit_num, k) * (ber ** k) * ((1 - ber) ** (self.bit_num - k)) for k in range(self.bit_num + 1)])
        error_probs = torch.tensor([(ber ** k) * ((1 - ber) ** (self.bit_num - k)) for k in range(self.bit_num + 1)])

        error_bits = torch.sum(symbol_bits.unsqueeze(1) ^ symbol_bits.unsqueeze(0), dim=2)

        transition_matrix = error_probs[error_bits]
        self.transition_matrix = transition_matrix
        self.transition_matrix = self.transition_matrix.to(self.device)
        self.transition_matrix.requires_grad = True
        self.ber = ber
        return transition_matrix, self.ber

    @staticmethod
    def comb(n, k):
        return torch.prod(torch.arange(n - k + 1, n + 1)) / torch.prod(torch.arange(1, k + 1))

    def BSC_forward(self, input_symbols, ber):

        if self.transition_matrix is None or ber != self.ber:
            self.calculate_transition_matrix(ber)

        transition_matrix = self.transition_matrix
        B, L = input_symbols.shape

        indices = input_symbols.view(-1)
        transition_probs = transition_matrix[indices]

        output_symbols = torch.multinomial(transition_probs, num_samples=1).view(B, L)

        return output_symbols
    def BSC_bit_flip(self, input_symbols, ber):
        if self.transition_matrix is None or ber != self.ber:
            self.calculate_transition_matrix(ber)
        B, L = input_symbols.shape
        input_device = input_symbols.device
        bit_stream = input_symbols.unsqueeze(2).bitwise_and(1 << torch.arange(self.bit_num, device= input_device)).ne(0).long().flatten()
        bernoulli_tensor = torch.bernoulli(torch.tensor([ber for i in range(len(bit_stream))], device= input_device))
        flipped_stream = bit_stream ^ bernoulli_tensor.long()

        out_stream = torch.tensor(flipped_stream, dtype=torch.float32, device=input_device).reshape(-1,
                                                                                                   self.bit_num)
        out = torch.matmul(out_stream, self.weights).reshape(B,L).long()

        return out



