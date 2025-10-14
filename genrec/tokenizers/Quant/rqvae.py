import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.init import xavier_normal_ 
from torch.nn.init import xavier_normal_
from sklearn.cluster import KMeans
from k_means_constrained import KMeansConstrained

class RQVAE(nn.Module):
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
                 ):
        super(RQVAE, self).__init__()
        self.in_dim = in_dim
        self.num_emb_list = num_emb_list
        self.e_dim = e_dim
        self.layers = layers
        self.dropout_prob = dropout_prob
        self.bn = bn
        self.loss_type = loss_type
        self.quant_loss_weight = quant_loss_weight
        self.kmeans_init = kmeans_init
        self.kmeans_iters = kmeans_iters
        self.commitment_beta = commitment_beta
        self.encode_layer_dims = [self.in_dim] + self.layers + [self.e_dim]
        self.encoder = MLPLayers(layers=self.encode_layer_dims,
                                 dropout=self.dropout_prob, bn=self.bn)

        self.rq = ResidualVectorQuantizer(num_emb_list, e_dim,
                                          commitment_beta=self.commitment_beta,
                                          kmeans_init=self.kmeans_init,
                                          kmeans_iters=self.kmeans_iters,
                                          )

        self.decode_layer_dims = self.encode_layer_dims[::-1]
        self.decoder = MLPLayers(layers=self.decode_layer_dims,
                                 dropout=self.dropout_prob, bn=self.bn)

    def forward(self, x): 
        x = self.encoder(x)
        x_q, rq_loss, indices = self.rq(x)
        out = self.decoder(x_q)

        return out, rq_loss, indices, x_q

    def vq_initialization(self, x):
        self.rq.vq_ini(self.encoder(x))

    @torch.no_grad()
    def get_indices(self, xs):
        x_e = self.encoder(xs)
        _, _, indices = self.rq(x_e)
        return indices

    def compute_loss(self, out, quant_loss, dense_out, xs=None):
        if self.loss_type == 'mse':
            loss_recon = F.mse_loss(out, xs, reduction='mean')
        elif self.loss_type == 'l1':
            loss_recon = F.l1_loss(out, xs, reduction='mean')
        else:
            raise ValueError('incompatible loss type')

        total_loss = loss_recon + self.quant_loss_weight * quant_loss

        return total_loss, loss_recon, quant_loss

def activation_layer(activation_name="relu", emb_dim=None):
    if activation_name is None:
        activation = None
    elif isinstance(activation_name, str):
        if activation_name.lower() == "sigmoid":
            activation = nn.Sigmoid()
        elif activation_name.lower() == "tanh":
            activation = nn.Tanh()
        elif activation_name.lower() == "relu":
            activation = nn.ReLU()
        elif activation_name.lower() == "leakyrelu":
            activation = nn.LeakyReLU()
        elif activation_name.lower() == "none":
            activation = None
    elif issubclass(activation_name, nn.Module):
        activation = activation_name()
    else:
        raise NotImplementedError(
            "activation function {} is not implemented".format(activation_name)
        )
    return activation

def kmeans(samples, num_clusters, num_iters=10):
    B, dim, dtype, device = samples.shape[0], samples.shape[-1], samples.dtype, samples.device
    x = samples.cpu().detach().numpy()
    cluster = KMeans(n_clusters=num_clusters, max_iter=num_iters, n_init=10).fit(x)
    centers = cluster.cluster_centers_
    tensor_centers = torch.from_numpy(centers).to(device)
    return tensor_centers

class MLPLayers(nn.Module):
    def __init__(self, layers, dropout=0.0, activation="relu", bn=False):
        super(MLPLayers, self).__init__()
        self.layers = layers
        self.dropout = dropout
        self.activation = activation
        self.use_bn = bn

        mlp_modules = []
        for idx, (input_size, output_size) in enumerate(
                zip(self.layers[:-1], self.layers[1:])
        ):
            mlp_modules.append(nn.Dropout(p=self.dropout))
            mlp_modules.append(nn.Linear(input_size, output_size))
            if self.use_bn:
                mlp_modules.append(nn.BatchNorm1d(num_features=output_size))
            activation_func = activation_layer(self.activation, output_size)
            if activation_func is not None and idx != (len(self.layers) - 2):
                mlp_modules.append(activation_func)

        self.mlp_layers = nn.Sequential(*mlp_modules)
        self.apply(self.init_weights)

    def init_weights(self, module):
        if isinstance(module, nn.Linear):
            xavier_normal_(module.weight.data)
            if module.bias is not None:
                module.bias.data.fill_(0.0)

    def forward(self, input_feature):
        return self.mlp_layers(input_feature)

class ResidualVectorQuantizer(nn.Module):
    def __init__(self, n_e_list, e_dim,commitment_beta=0.25,
                 kmeans_init=False, kmeans_iters=100):
        super().__init__()
        self.n_e_list = n_e_list
        self.e_dim = e_dim
        self.num_quantizers = len(n_e_list)
        self.kmeans_init = kmeans_init
        self.kmeans_iters = kmeans_iters
        self.commitment_beta = commitment_beta
        self.vq_layers = nn.ModuleList([VectorQuantizer(n_e, e_dim,
                                                        mu = self.commitment_beta,
                                                        kmeans_init=self.kmeans_init,
                                                        kmeans_iters=self.kmeans_iters
                                                        )
                                        for n_e in n_e_list])

    def get_codebook(self):
        all_codebook = []
        for quantizer in self.vq_layers:
            codebook = quantizer.get_codebook()
            all_codebook.append(codebook)
        return torch.stack(all_codebook)

    def vq_ini(self, x):
        x_q = 0
        residual = x
        for idx, quantizer in enumerate(self.vq_layers):
            x_res = quantizer.vq_init(residual)
            residual = residual - x_res
            x_q = x_q + x_res

    def forward(self, x):
        all_losses = []
        all_indices = []
        x_q = 0
        residual = x

        for idx, quantizer in enumerate(self.vq_layers):
            x_res, loss, indices = quantizer(residual, idx)
            residual = residual - x_res
            x_q = x_q + x_res
            all_losses.append(loss)
            all_indices.append(indices)

        mean_losses = torch.stack(all_losses).mean()
        all_indices = torch.stack(all_indices, dim=-1)

        return x_q, mean_losses, all_indices

class VectorQuantizer(nn.Module):
    def __init__(self, n_e, e_dim, mu=0.25,
                 kmeans_init=False, kmeans_iters=10):
        super().__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.mu = mu
        self.kmeans_init = kmeans_init
        self.kmeans_iters = kmeans_iters

        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        if not kmeans_init:
            self.initted = True
            self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)
        else:
            self.initted = False
            self.embedding.weight.data.zero_()

    def get_codebook(self):
        return self.embedding.weight

    def get_codebook_entry(self, indices, shape=None):
        z_q = self.embedding(indices)
        if shape is not None:
            z_q = z_q.view(shape)
        return z_q

    def init_emb(self, data):
        centers = kmeans(
            data,
            self.n_e,
            self.kmeans_iters,
        )
        self.embedding.weight.data.copy_(centers)
        self.initted = True


    @staticmethod
    def center_distance_for_constraint(distances):
        max_distance = distances.max()
        min_distance = distances.min()
        middle = (max_distance + min_distance) / 2
        amplitude = max_distance - middle + 1e-5
        assert amplitude > 0
        centered_distances = (distances - middle) / amplitude
        return centered_distances

    def vq_init(self, x):
        latent = x.view(-1, self.e_dim)
        if not self.initted:
            self.init_emb(latent)
        _distance_flag = 'distance'
        if _distance_flag == 'distance':
            d = torch.sum(latent ** 2, dim=1, keepdim=True) + \
                torch.sum(self.embedding.weight ** 2, dim=1, keepdim=True).t() - \
                2 * torch.matmul(latent, self.embedding.weight.t())
        else:
            d = latent @ self.embedding.weight.t()
        if _distance_flag == 'distance':
            indices = torch.argmin(d, dim=-1)
        else:
            indices = torch.argmax(d, dim=-1)
        x_q = self.embedding(indices).view(x.shape)
        return x_q

    def forward(self, x, idx):
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
        commitment_loss = F.mse_loss(x_q.detach(), x)
        codebook_loss = F.mse_loss(x_q, x.detach())
        loss = codebook_loss + self.mu * commitment_loss
        x_q = x + (x_q - x).detach()
        indices = indices.view(x.shape[:-1])
        return x_q, loss, indices