import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.init import xavier_normal_ 
from torch.nn.init import xavier_normal_
from sklearn.cluster import KMeans
from k_means_constrained import KMeansConstrained
from .rqvae import RQVAE, ResidualVectorQuantizer, VectorQuantizer
import random
class LETTERRQVAE(RQVAE):
    def __init__(self,
                 in_dim=768,
                 num_emb_list=None,
                 e_dim=32,
                 layers=None,
                 dropout_prob=0.0,
                 bn=False,
                 loss_type="mse",
                 quant_loss_weight=1.0,
                 kmeans_init=False,
                 kmeans_iters=100,
                 commitment_beta=0.25,
                 diversity_beta=1,
                 ):
        super().__init__(
            in_dim=in_dim,
            num_emb_list=num_emb_list,
            e_dim=e_dim,
            layers=layers,
            dropout_prob=dropout_prob,
            bn=bn,
            loss_type=loss_type,
            quant_loss_weight=quant_loss_weight,
            kmeans_init=kmeans_init,
            kmeans_iters=kmeans_iters,
            commitment_beta=commitment_beta,
        )
        self.diversity_beta = diversity_beta

        self.rq = LETTERResidualVectorQuantizer(num_emb_list, e_dim,
                                          commitment_beta=self.commitment_beta,
                                          diversity_beta=self.diversity_beta,
                                          kmeans_init=self.kmeans_init,
                                          kmeans_iters=self.kmeans_iters,
                                          )
    def forward(self, x, labels): 
        x = self.encoder(x)
        x_q, rq_loss, indices = self.rq(x, labels)
        out = self.decoder(x_q)

        return out, rq_loss, indices, x_q
    @torch.no_grad()
    def get_indices(self, xs, labels):
        x_e = self.encoder(xs)
        _, _, indices = self.rq(x_e, labels)
        return indices
class LETTERResidualVectorQuantizer(ResidualVectorQuantizer):
    def __init__(self, n_e_list, e_dim,commitment_beta=0.25,
                 diversity_beta=1,kmeans_init=False, kmeans_iters=100):
        super().__init__(
            n_e_list,
            e_dim,
            commitment_beta=commitment_beta,
            kmeans_init=kmeans_init,
            kmeans_iters=kmeans_iters,
        )
        self.diversity_beta = diversity_beta
        self.vq_layers = nn.ModuleList([LETTERVectorQuantizer(n_e, e_dim,
                                                        mu = self.commitment_beta,
                                                        diversity_beta=self.diversity_beta,
                                                        kmeans_init=self.kmeans_init,
                                                        kmeans_iters=self.kmeans_iters
                                                        )
                                        for n_e in n_e_list])
    def forward(self, x, labels):
        all_losses = []
        all_indices = []
        x_q = 0
        residual = x

        for idx, quantizer in enumerate(self.vq_layers):
            if labels is not None:
                label = labels[str(idx)]
            else:
                label = None
            x_res, loss, indices = quantizer(residual, label, idx)
            residual = residual - x_res
            x_q = x_q + x_res
            all_losses.append(loss)
            all_indices.append(indices)

        mean_losses = torch.stack(all_losses).mean()
        all_indices = torch.stack(all_indices, dim=-1)

        return x_q, mean_losses, all_indices
class LETTERVectorQuantizer(VectorQuantizer):
    def __init__(self, n_e, e_dim, mu=0.25, diversity_beta=1,
                 kmeans_init=False, kmeans_iters=10):
        super().__init__(
            n_e,
            e_dim,
            mu=mu,
            kmeans_init=kmeans_init,
            kmeans_iters=kmeans_iters,
        )
        self.diversity_beta = diversity_beta

    def init_emb(self, data):
        centers, _ = self.constrained_km(data, self.n_e)
        self.embedding.weight.data.copy_(centers)
        self.initted = True
    def constrained_km(self, data, n_clusters=10):
        x = data.cpu().detach().numpy()
        size_min = min(len(data) // (n_clusters * 2), 50)
        clf = KMeansConstrained(n_clusters=n_clusters, size_min=size_min, size_max=size_min * 4, max_iter=10, n_init=10,
                                n_jobs=10, verbose=False)
        clf.fit(x)
        t_centers = torch.from_numpy(clf.cluster_centers_)
        t_labels = torch.from_numpy(clf.labels_).tolist()
        return t_centers, t_labels

    def diversity_loss(self, x_q, indices, labels):
        # 1. 确保 labels 是 Tensor 且在正确的设备上
        if not isinstance(labels, torch.Tensor):
            labels_tensor = torch.tensor(labels, device=x_q.device)
        else:
            labels_tensor = labels.to(x_q.device)

        # 2. 获取当前 batch 样本所属的 cluster
        current_cluster_ids = labels_tensor[indices] # [B]

        # 3. 构造同类掩码 (Batch_Size, Num_Embeddings)
        # 标记出哪些 embedding 属于同一类
        mask = (current_cluster_ids.unsqueeze(1) == labels_tensor.unsqueeze(0)) 
        
        # 4. 排除自身索引
        self_mask = torch.zeros_like(mask)
        self_mask.scatter_(1, indices.unsqueeze(1), True)
        mask = mask & ~self_mask

        # 5. 随机采样正样本索引 (在 GPU 上并行)
        # 增加极小值防止全 0 行报错
        weights = mask.float() + 1e-12
        pos_sample_indices = torch.multinomial(weights, 1).squeeze(1)

        # 6. 计算相似度
        emb = self.embedding.weight
        sim = torch.matmul(x_q, emb.t()) # [B, N_E]

        # 7. 屏蔽掉自身的得分 (减去大值)
        sim.scatter_(1, indices.unsqueeze(1), -1e9)

        # 8. Cross Entropy
        loss = F.cross_entropy(sim, pos_sample_indices)
        return loss

    def forward(self, x, label, idx):
        latent = x.view(-1, self.e_dim)
        if not self.initted and self.training:
            self.init_emb(latent)
        _distance_flag = 'distance'
        if _distance_flag == 'distance':
            d = torch.sum(latent ** 2, dim=1, keepdim=True) + \
                torch.sum(self.embedding.weight ** 2, dim=1, keepdim=True).t() - \
                2 * torch.matmul(latent, self.embedding.weight.t())
        else:
            d = latent @ self.embedding.weight.t()
        if _distance_flag == 'distance':
            if idx != -1:
                indices = torch.argmin(d, dim=-1)
            else:
                temp = 1.0
                prob_dist = F.softmax(-d / temp, dim=1)
                indices = torch.multinomial(prob_dist, 1).squeeze()
        else:
            indices = torch.argmax(d, dim=-1)

        x_q = self.embedding(indices).view(x.shape)

        if label is not None:
            diversity_loss = self.diversity_loss(x_q, indices, label)
        else:
            diversity_loss = torch.tensor(0.0, device=x.device)
        commitment_loss = F.mse_loss(x_q.detach(), x)
        codebook_loss = F.mse_loss(x_q, x.detach())
        loss = codebook_loss + self.mu * commitment_loss + self.diversity_beta * diversity_loss
        x_q = x + (x_q - x).detach()
        indices = indices.view(x.shape[:-1])
        return x_q, loss, indices