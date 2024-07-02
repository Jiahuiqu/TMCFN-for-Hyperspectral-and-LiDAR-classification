import torch
import torch.nn as nn
from DGMlib.model_dDGM_trento import DGM_Model
import copy
import torch.nn.functional as F
from collections import OrderedDict
import clip
import numpy as np
import en_transformer
import utils
from einops import rearrange, repeat


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1):
        super().__init__()

        # all conv layers have stride 1. an avgpool is performed after the second convolution when stride > 1
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu2 = nn.ReLU(inplace=True)

        self.avgpool = nn.AvgPool2d(stride) if stride > 1 else nn.Identity()

        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu3 = nn.ReLU(inplace=True)

        self.downsample = None
        self.stride = stride

        if stride > 1 or inplanes != planes * Bottleneck.expansion:
            # downsampling layer is prepended with an avgpool, and the subsequent convolution has stride 1
            self.downsample = nn.Sequential(OrderedDict([
                ("-1", nn.AvgPool2d(stride)),
                ("0", nn.Conv2d(inplanes, planes * self.expansion, 1, stride=1, bias=False)),
                ("1", nn.BatchNorm2d(planes * self.expansion))
            ]))

    def forward(self, x: torch.Tensor):
        identity = x

        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.relu2(self.bn2(self.conv2(out)))
        out = self.avgpool(out)
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu3(out)
        return out

class AttentionPool2d(nn.Module):
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.randn(spacial_dim ** 2 + 1, embed_dim) / embed_dim ** 0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, x):
        x = x.flatten(start_dim=2).permute(2, 0, 1)  # NCHW -> (HW)NC
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
        x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC
        x, _ = F.multi_head_attention_forward(
            query=x[:1], key=x, value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False
        )
        return x.squeeze(0)

class ModifiedResNet(nn.Module):
    """
    A ResNet class that is similar to torchvision's but contains the following changes:
    - There are now 3 "stem" convolutions as opposed to 1, with an average pool instead of a max pool.
    - Performs anti-aliasing strided convolutions, where an avgpool is prepended to convolutions with stride > 1
    - The final pooling layer is a QKV attention instead of an average pool
    """

    def __init__(self, layers, output_dim, heads, input_resolution=224, width=64):
        super().__init__()
        self.output_dim = output_dim
        self.input_resolution = input_resolution

        # the 3-layer stem
        self.conv1 = nn.Conv2d(3, width // 2, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width // 2)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(width // 2, width // 2, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(width // 2)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(width // 2, width, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(width)
        self.relu3 = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(2)

        # residual layers
        self._inplanes = width  # this is a *mutable* variable used during construction
        self.layer1 = self._make_layer(width, layers[0])
        self.layer2 = self._make_layer(width * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(width * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(width * 8, layers[3], stride=2)

        embed_dim = width * 32  # the ResNet feature dimension
        self.attnpool = AttentionPool2d(input_resolution // 32, embed_dim, heads, output_dim)

    def _make_layer(self, planes, blocks, stride=1):
        layers = [Bottleneck(self._inplanes, planes, stride)]

        self._inplanes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self._inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        def stem(x):
            x = self.relu1(self.bn1(self.conv1(x)))
            x = self.relu2(self.bn2(self.conv2(x)))
            x = self.relu3(self.bn3(self.conv3(x)))
            x = self.avgpool(x)
            return x

        x = x.type(self.conv1.weight.dtype)
        x = stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.attnpool(x)

        return x

class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)

class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)

class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)

class VisionTransformer(nn.Module):
    def __init__(self, img_size, input_channels, patch_size, dim, depth, heads, mlp_dim, dropout, dim_head, pool='cls'):
        super(VisionTransformer, self).__init__()

        num_patches = (img_size // patch_size) ** 2
        patch_dim = input_channels * patch_size ** 2

        assert pool in {'cls', 'mean'}
        self.pool = pool

        self.patch_embedding = nn.Conv2d(input_channels, dim, kernel_size=patch_size, stride=patch_size)
        self.positional_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(dropout)
        self.to_cls_token = nn.Identity()

        self.transformer_encoder = en_transformer.Transformer(dim=dim, depth=depth, heads=heads, dim_head=dim_head,
                                                              mlp_head=mlp_dim, dropout=dropout,
                                                              num_channel=num_patches)

    def forward(self, data_HSI):
        data_HSI = self.patch_embedding(data_HSI)  # (batch_size, dim, num_patches_h, num_patches_w)
        data_HSI = rearrange(data_HSI, 'b d h w -> b (h w) d')  # (batch_size, num_patches, dim)
        b, n, _ = data_HSI.shape

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)
        data_HSI = torch.cat((cls_tokens, data_HSI), dim=1)
        positional_embedding = self.positional_embedding[:, :(n+1)]
        data_HSI = data_HSI + positional_embedding
        data_HSI = self.dropout(data_HSI)

        data_HSI = self.transformer_encoder(data_HSI)
        # global average pooling or cls
        data_HSI = data_HSI.mean(dim=1) if self.pool == 'mean' else data_HSI[:, 0]

        return data_HSI

# loss fn
def loss_fn(x, y):
    x = F.normalize(x, dim=-1, p=2)
    y = F.normalize(y, dim=-1, p=2)
    return 2 - 2 * (x * y).sum(dim=-1)

def set_requires_grad(model, bool_val):
    for p in model.parameters():
        p.requires_grad = bool_val

def update_moving_average(ema_updater, ma_model, current_model):
    for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
        old_weight, up_weight = ma_params.data, current_params.data
        ma_params.data = ema_updater.update_average(old_weight, up_weight)

# exponential moving average
class EMA():
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new
##
class Attention_Feature_Fusion_Module(torch.nn.Module):
    def __init__(self):
        super(Attention_Feature_Fusion_Module, self).__init__()
        self.MLP_1 = torch.nn.Sequential(nn.Linear(32, 128),
                                         nn.BatchNorm1d(128),
                                         nn.ReLU(),
                                         nn.Linear(128, 64),
                                         )
        self.MLP_2 = torch.nn.Sequential(nn.Linear(32, 128),
                                         nn.BatchNorm1d(128),
                                         nn.ReLU(),
                                         nn.Linear(128, 64),
                                         )
        self.MLP_3 = torch.nn.Sequential(nn.Linear(32 *  3, 128),
                                         nn.BatchNorm1d(128),
                                         nn.ReLU(),
                                         nn.Linear(128, 32))
        self.sigmoid = nn.Sigmoid()


    def forward(self, x1, x2):
        x1_ = torch.unsqueeze(self.MLP_1(torch.squeeze(x1)), dim=1) # N * 1 * C
        x2_ = torch.unsqueeze(self.MLP_2(torch.squeeze(x2)), dim=-1)  #  N * C * 1
        alpha = torch.squeeze(self.sigmoid(torch.matmul(x1_, x2_)), dim=-1)
        add_x = torch.squeeze(x1) * alpha + torch.squeeze(x2) * (1 - alpha)
        out = self.MLP_3(torch.cat([x1, add_x, x2], dim=-1))
        return out


class Contrastive_Fusion_Feature_Learning_Module_ViT(torch.nn.Module):
    def __init__(self, img_size, input_channels, patch_size, dim, depth, heads, mlp_dim,
                                             dropout, dim_head, moving_average_decay=0.99, use_momentum=True):
        super(Contrastive_Fusion_Feature_Learning_Module_ViT, self).__init__()
        self.ViT_Encoder_online = VisionTransformer(img_size, input_channels, patch_size, dim, depth, heads, mlp_dim,
                                             dropout, dim_head)
        self.AFFM_online = Attention_Feature_Fusion_Module()
        self.projector_online = torch.nn.Sequential(nn.Linear(32, 512),
                                                  nn.BatchNorm1d(512),
                                                  nn.ReLU(),
                                                nn.Linear(512, 256))
        self.ViT_Encoder_target, self.AFFM_target, self.projector_target = self._get_target_encoder_and_Fusion_and_projection()

        self.predictor_online = torch.nn.Sequential(nn.Linear(256, 512),
                                                    nn.BatchNorm1d(512),
                                                    nn.ReLU(),
                                                    nn.Linear(512, 256))

        self.use_momentum = use_momentum
        self.target_ema_updater = EMA(moving_average_decay)


    def _get_target_encoder_and_Fusion_and_projection(self):
        target_encoder = copy.deepcopy(self.ViT_Encoder_online)
        AFFM_target = copy.deepcopy(self.AFFM_online)
        target_projection = copy.deepcopy(self.projector_online)
        set_requires_grad(target_encoder, False)
        set_requires_grad(AFFM_target, False)
        set_requires_grad(target_projection, False)
        return target_encoder, AFFM_target, target_projection

    def update_moving_average(self):
        assert self.use_momentum, 'you do not need to update the moving average, since you have turned off momentum for the target encoder'
        assert self.ViT_Encoder_target is not None, 'target encoder has not been created yet'
        update_moving_average(self.target_ema_updater, self.ViT_Encoder_target, self.ViT_Encoder_online)
        update_moving_average(self.target_ema_updater, self.AFFM_target, self.AFFM_online)
        update_moving_average(self.target_ema_updater, self.projector_target, self.projector_online)


    def forward(self, x1, x2, x3):
        x1_online = self.ViT_Encoder_online(x1)
        x2_online = self.ViT_Encoder_online(x2)
        x3_online = self.ViT_Encoder_online(x3)
        x1_target = self.ViT_Encoder_target(x1)
        x2_target = self.ViT_Encoder_target(x2)
        x3_target = self.ViT_Encoder_target(x3)

        o1 = self.AFFM_online(x1_online, x1_online)
        o2 = self.AFFM_online(x2_online, x2_online)
        o3 = self.AFFM_online(x3_online, x3_online)

        t1 = self.AFFM_target(x1_target, x2_target)
        t2 = self.AFFM_target(x1_target, x3_target)
        t3 = self.AFFM_target(x2_target, x3_target)

        loss1 = loss_fn(self.predictor_online(self.projector_online(o1)), self.projector_target(t3).detach())
        loss2 = loss_fn(self.predictor_online(self.projector_online(o2)), self.projector_target(t2).detach())
        loss3 = loss_fn(self.predictor_online(self.projector_online(o3)), self.projector_target(t1).detach())
        loss = (loss1 + loss2 + loss3).mean()

        return o1, o2, o3, loss

##
class Text_Guided_Attention_Feature_Fusion_Module(torch.nn.Module):

    def __init__(self):
        super(Text_Guided_Attention_Feature_Fusion_Module, self).__init__()
        self.MLP_Text = torch.nn.Sequential(nn.Linear(512, 32))
        self.MLP_Image = torch.nn.Sequential(nn.Linear(32, 32))
        self.linear_mapping_matching_dim = nn.Linear(512, 32 * 3)
        self.sig = nn.Sigmoid()

    def forward(self, x1, x2, x3, ft):

        ft_ = self.linear_mapping_matching_dim(ft)
        x = torch.cat([x1, x2, x3], dim=1)
        S = torch.matmul(x, torch.transpose(ft_, dim0=1, dim1=0))
        text_vector_weights = torch.nn.functional.softmax(S, dim=1)

        ft_weighted = torch.matmul(text_vector_weights, ft)
        e1 = self.sig(self.MLP_Image(x1) + self.MLP_Text(ft_weighted))
        e2 = self.sig(self.MLP_Image(x2) + self.MLP_Text(ft_weighted))
        e3 = self.sig(self.MLP_Image(x3) + self.MLP_Text(ft_weighted))
        image_weights = torch.nn.functional.softmax(torch.cat([e1, e2, e3], dim=1), dim=1) # N * 3
        Z = torch.squeeze(torch.matmul(torch.unsqueeze(torch.unsqueeze(image_weights[:,0], dim=-1), dim=1), torch.unsqueeze(x1, dim=1))) + \
            torch.squeeze(torch.matmul(torch.unsqueeze(torch.unsqueeze(image_weights[:,1], dim=-1), dim=1), torch.unsqueeze(x2, dim=1))) + \
            torch.squeeze(torch.matmul(torch.unsqueeze(torch.unsqueeze(image_weights[:,2], dim=-1), dim=1), torch.unsqueeze(x3, dim=1)))
        return Z

##
class Vision_Net_Final(nn.Module):
    def __init__(self, hparams, patchsize_HSI, device1, device2, class_num):
        super(Vision_Net_Final, self).__init__()
        self.device1 = device1
        self.device2 = device2
        self.hparams = hparams
        # self.FCFLM = Contrastive_Fusion_Feature_Learning_Module_ViT(img_size=patchsize_HSI, input_channels=32,patch_size=2, dim=32, depth=2, heads=4, mlp_dim=32, dropout=0.3, dim_head=32).to(self.device1)
        self.FCFLM = Contrastive_Fusion_Feature_Learning_Module_ViT(img_size=patchsize_HSI, input_channels=32,patch_size=2, dim=32, depth=3, heads=4, mlp_dim=32, dropout=0.3, dim_head=32).to(self.device1)
        self.Graph_DGFL = DGM_Model(hparams).to(self.device2)  # graph dynamic construction and embedding
        self.TAF2M = Text_Guided_Attention_Feature_Fusion_Module().to(self.device2)
        self.projector = nn.Sequential(nn.Linear(32, 512)).to(self.device2)  # dimension matching

    def forward(self,patch_hsi=None, patch_lidar=None, ft=None, labeled_mask=None, warm_up=False, len_labeled=None):

        x1, x2, x3,loss_cl = self.FCFLM(patch_hsi, patch_lidar+patch_hsi, patch_lidar)

        if(len_labeled == None):
            x1 = x1[labeled_mask]
            x2 = x2[labeled_mask]
            x3 = x3[labeled_mask]
        else:
            x1 = x1[0:len_labeled, :][labeled_mask]
            x2 = x2[0:len_labeled, :][labeled_mask]
            x3 = x3[0:len_labeled, :][labeled_mask]

        x1 = x1.to(self.device2)
        x2 = x2.to(self.device2)
        x3 = x3.to(self.device2)
        loss_cl = loss_cl.to(self.device2)

        init_edge = utils.get_adj(x1.detach(), x2.detach(), x3.detach(), self.device1, self.hparams.k).to(self.device2)

        x1 = torch.unsqueeze(x1, dim=0)
        x2 = torch.unsqueeze(x2, dim=0)
        x3 = torch.unsqueeze(x3, dim=0)

        x1, x2, x3, edges, logprobs = self.Graph_DGFL(x1, x2, x3, init_edge,warm_up=warm_up)

        x1 = torch.squeeze(x1, dim=0)
        x2 = torch.squeeze(x2, dim=0)
        x3 = torch.squeeze(x3, dim=0)

        Z_image = self.TAF2M(x1, x2, x3, ft)

        Z_vison = self.projector(Z_image)

        return Z_vison, loss_cl, logprobs

##
class Text_Net(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 context_length: int,
                 vocab_size: int,
                 transformer_width: int,
                 transformer_heads: int,
                 transformer_layers: int
                 ):
        super().__init__()

        self.context_length = context_length

        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask()
        )

        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
        self.ln_final = LayerNorm(transformer_width)
        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.initialize_parameters()

    def initialize_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)
        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)
        if self.text_projection is not None:
            nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)

    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    def forward(self, text):
        x = self.token_embedding(text).type(torch.float32)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding.type(torch.float32)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)  # torch.Size([77, 6, 512])
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(torch.float32)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection

        return x

class Model_All(torch.nn.Module):
    def __init__(self, hparams,  patchsize_HSI, device1, device2, class_num, text):
        super(Model_All, self).__init__()
        self.device = device1
        self.device1 = device1
        self.device2 = device2
        self.vision_net = Vision_Net_Final(hparams,  patchsize_HSI, device1=self.device1, device2=self.device2 , class_num=class_num)
        self.text_net = Text_Net(embed_dim=512,
                                 context_length=77,
                                 vocab_size=49408,
                                 transformer_width=512,
                                 transformer_heads=8,
                                 transformer_layers=12)
        self.text_net.load_state_dict(torch.load("./text_encoder.pth", map_location='cpu'))  ##
        with torch.no_grad():
            text = clip.tokenize(text)
            self.feature_text = self.text_net(text).to(self.device2)
        del self.text_net

    def forward(self, patch_hsi, patch_lidar, labeled_mask, warm_up, len_labeled=None):

        Z_vison, loss_vision_CL, logprobs = self.vision_net(patch_hsi=torch.squeeze(patch_hsi).to(self.device1),
                                                            patch_lidar=torch.squeeze(patch_lidar).to(self.device1),
                                                            ft=self.feature_text,
                                                            labeled_mask=labeled_mask, warm_up=warm_up,
                                                            len_labeled=len_labeled)

        Z_text = self.feature_text

        return Z_vison, Z_text, loss_vision_CL,logprobs
