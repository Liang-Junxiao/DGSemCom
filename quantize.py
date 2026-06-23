import torch
from torch import nn
import torch.nn.functional as F
class PQuantize(nn.Module):
    def __init__(self, dim, n_embed, chunk_num, **kwargs):
        super().__init__()
        self.codebook_list = nn.ModuleList()
        self.dim = dim
        self.n_embed = n_embed
        self.chunk_num = chunk_num
        self.share_codebook = kwargs['share_codebook']
        self.config = kwargs['config']
        for i in range(self.chunk_num):
            quant = Quantize( int(dim / self.chunk_num), n_embed, config=self.config )
            self.codebook_list.append(quant)

    def forward(self, input, **kwargs):
        chunks = torch.chunk(input, chunks = self.chunk_num, dim = 1)
        quant_list = []
        diff_total = 0
        embed_id_list = []
        id_prob_list = []
        for id, chunk in enumerate(chunks):
            if self.config.norm_codebook is not None and self.config.norm_codebook == False:
                pass
            else:
                chunk = my_norm(chunk, 1, dim=1)
            if self.share_codebook == True:
                quant, diff, embed_id, id_prob = self.codebook_list[0](chunk)
            else:
                quant, diff, embed_id, id_prob = self.codebook_list[id](chunk)
            quant_list.append(quant)
            diff_total += diff
            embed_id_list.append(embed_id.unsqueeze(3))
            id_prob_list.append(id_prob)
        qunatize = torch.cat(tuple(quant_list), dim=1)
        embed_id = torch.cat(tuple(embed_id_list), dim=3)
        return qunatize, diff_total / len(chunks), embed_id, id_prob_list

    def de_quantize(self, id_prob_list):
        de_quantize_list = []
        for id, item in enumerate(id_prob_list):
            if self.share_codebook == True:
                de_quantize_list.append(self.codebook_list[0](item))
            else:
                de_quantize_list.append(self.codebook_list[id](item))
        de_quantize = torch.cat(*de_quantize_list, dim=1)
        return de_quantize

    def de_quantize_code(self, code):
        de_quantize_list = []
        for id in range(len(code[0][0][0])):
            if self.share_codebook == True:
                de_quantize_list.append(self.codebook_list[0].embed_code(code[:,:,:,id]))
            else:
                de_quantize_list.append(self.codebook_list[id].embed_code(code[:,:,:,id]))
        de_quantize = torch.cat(de_quantize_list, dim=3)
        return de_quantize.permute(0,3,1,2)


class Quantize(nn.Module):
    def __init__(self, dim, n_embed, **kwargs):
        super().__init__()

        self.dim = dim
        self.n_embed = n_embed
        self.embed = nn.Parameter(torch.randn(dim, n_embed))
        nn.init.normal(self.embed)
        self.embed.data = F.normalize(self.embed.data, p=2, dim=0)
        self.cross_entropy = nn.CrossEntropyLoss()
        self.tau = 0.01
        self.config = kwargs['config']
    def forward(self, input):
        input = input.permute(0, 2, 3, 1).contiguous()  # [B x D x H x W] -> [B x H x W x D]
        if self.config.norm_codebook is not None and self.config.norm_codebook == False:
            pass
        else:
            self.normalize_codebook()
        flatten = input.reshape(-1, self.dim)

        if self.config.norm_codebook is not None and self.config.norm_codebook == False:
            dist = flatten.pow(2).sum(1, keepdim=True)- 2 * flatten @ self.embed + self.embed.pow(2).sum(0, keepdim=True)
        else:
            dist = - 2 * flatten @ self.embed
        id_prob = F.softmax(-dist/self.tau , dim=1)
        id_prob = id_prob.view([*input.shape[:-1], self.n_embed])
        _, embed_ind = (-dist).max(1)

        embed_ind = embed_ind.view(*input.shape[:-1])

        quantize = self.embed_code(embed_ind)


        diff = 0.25 * (quantize.detach() - input).pow(2).mean() + (quantize -input.detach()).pow(2).mean()

        quantize = input + (quantize - input).detach()

        return quantize.permute(0, 3, 1, 2), diff, embed_ind, id_prob

    def embed_code(self, embed_id):
        return F.embedding(embed_id, self.embed.transpose(0, 1))

    def de_quantize(self, id_prob):
        max_values, _ = torch.max(id_prob, dim=3, keepdim=True)
        mask = torch.eq(id_prob, max_values).type_as(id_prob)
        id_prob = F.normalize(id_prob* mask, p=2, dim=3)
        # id_prob = id_prob* mask
        return (id_prob @ self.embed.transpose(0, 1)).permute(0, 3, 1, 2)

    def normalize_codebook(self):
        self.embed.data = F.normalize(self.embed.data, p=2, dim=0)

def my_norm(input, n, dim):
    return n* F.normalize(input, p=2, dim=dim)