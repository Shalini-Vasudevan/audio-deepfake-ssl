"""Detectors.
 mode='last'   : SSL last layer -> mean pool -> linear                 (baseline)
 mode='wsum'   : learnable softmax-weighted sum over ALL SSL layers -> attentive-stats pooling (LAF)
 + adv=True    : codec/noise-adversarial head with gradient reversal (CoLA = Codec-invariant Layer-Attentive)
 mode='lfcc'   : LFCC + LCNN, no SSL                                  (non-SSL baseline)
"""
import torch, torch.nn as nn, torch.nn.functional as F


class GRL(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lam):
        ctx.lam = lam
        return x.view_as(x)

    @staticmethod
    def backward(ctx, g):
        return -ctx.lam * g, None


class SSLFront(nn.Module):
    def __init__(self, name, finetune_top=0):
        super().__init__()
        from transformers import AutoModel, WavLMConfig, WavLMModel
        if name == "tiny":   # offline smoke-test model (random init)
            cfg = WavLMConfig(hidden_size=64, num_hidden_layers=3, num_attention_heads=4, intermediate_size=128,
                              conv_dim=(32,) * 7, conv_stride=(5, 2, 2, 2, 2, 2, 2), conv_kernel=(10, 3, 3, 3, 3, 2, 2),
                              num_feat_extract_layers=7, num_buckets=32, max_bucket_distance=64,
                              num_conv_pos_embeddings=16, num_conv_pos_embedding_groups=4)
            self.m = WavLMModel(cfg)
        else:
            self.m = AutoModel.from_pretrained(name)
        self.m.config.apply_spec_augment = False
        self.dim = self.m.config.hidden_size
        self.n_layers = self.m.config.num_hidden_layers + 1
        for p in self.m.parameters():
            p.requires_grad = False
        self.finetune_top = finetune_top
        if finetune_top > 0:
            for l in self.m.encoder.layers[-finetune_top:]:
                for p in l.parameters():
                    p.requires_grad = True

    def forward(self, wav):
        wav = (wav - wav.mean(1, keepdim=True)) / (wav.std(1, keepdim=True) + 1e-7)
        ctx = torch.enable_grad() if self.finetune_top > 0 else torch.no_grad()
        with ctx:
            if self.finetune_top == 0:
                self.m.eval()
            hs = self.m(wav, output_hidden_states=True).hidden_states
        return torch.stack(hs, 0)  # (L, B, T, D)


class AttStatPool(nn.Module):
    def __init__(self, d, h=128):
        super().__init__()
        self.att = nn.Sequential(nn.Linear(d, h), nn.Tanh(), nn.Linear(h, 1))

    def forward(self, x):  # B,T,D
        w = torch.softmax(self.att(x), 1)
        mu = (w * x).sum(1)
        sd = ((w * (x - mu[:, None]) ** 2).sum(1)).clamp(min=1e-5).sqrt()
        return torch.cat([mu, sd], -1), w.squeeze(-1)


class LCNN(nn.Module):
    def __init__(self):
        super().__init__()
        import torchaudio
        self.lfcc = torchaudio.transforms.LFCC(16000, n_lfcc=60, n_filter=60,
                                               speckwargs=dict(n_fft=512, win_length=400, hop_length=160))

        def mfm_conv(i, o, k, s=1, p=0):
            return nn.Sequential(nn.Conv2d(i, 2 * o, k, s, p), nn.MaxPool3d((2, 1, 1)) if False else MFM())
        self.net = nn.Sequential(
            mfm_conv(1, 32, 5, 1, 2), nn.MaxPool2d(2), mfm_conv(32, 48, 1), mfm_conv(48, 64, 3, 1, 1), nn.MaxPool2d(2),
            mfm_conv(64, 64, 3, 1, 1), nn.MaxPool2d(2), mfm_conv(64, 32, 3, 1, 1), nn.AdaptiveAvgPool2d(1))
        self.fc = nn.Linear(32, 2)

    def forward(self, wav):
        x = self.lfcc(wav).unsqueeze(1)
        x = (x - x.mean((2, 3), keepdim=True)) / (x.std((2, 3), keepdim=True) + 1e-5)
        return self.fc(self.net(x).flatten(1)), None, None


class MFM(nn.Module):
    def forward(self, x):
        a, b = x.chunk(2, 1)
        return torch.max(a, b)


class Detector(nn.Module):
    def __init__(self, ssl="microsoft/wavlm-large", mode="wsum", adv=False, finetune_top=0, n_dom=3, cached_dims=None, uniform=False):
        super().__init__()
        self.mode, self.adv, self.lam, self.uniform = mode, adv, 0.0, uniform
        if mode == "lfcc":
            self.net = LCNN()
            return
        if cached_dims:            # (D, L): train the head on pre-extracted SSL features (CPU-friendly)
            D, L = cached_dims
        else:
            self.front = SSLFront(ssl, finetune_top)
            D, L = self.front.dim, self.front.n_layers
        self.layer_logits = nn.Parameter(torch.zeros(L))
        self.proj = nn.Sequential(nn.Linear(D, 256), nn.LayerNorm(256), nn.GELU(), nn.Dropout(0.1))
        if mode == "wsum":
            self.pool = AttStatPool(256)
            self.cls = nn.Sequential(nn.Linear(512, 128), nn.GELU(), nn.Dropout(0.2), nn.Linear(128, 2))
            feat = 512
        else:
            self.cls = nn.Linear(256, 2)
            feat = 256
        if adv:
            self.dom = nn.Sequential(nn.Linear(feat, 128), nn.GELU(), nn.Linear(128, n_dom))

    def layer_weights(self):
        if self.uniform:
            return torch.full_like(self.layer_logits, 1.0 / len(self.layer_logits))
        return torch.softmax(self.layer_logits, 0)

    def forward(self, wav):
        if self.mode == "lfcc":
            return self.net(wav)
        return self.head(self.front(wav))

    def head(self, hs):  # hs: L,B,T,D
        if self.mode == "last":
            x = self.proj(hs[-1])
            f, att = x.mean(1), None
        else:
            w = self.layer_weights().view(-1, 1, 1, 1).to(hs.dtype)
            x = self.proj((w * hs).sum(0))
            f, att = self.pool(x)
        logits = self.cls(f)
        dom = self.dom(GRL.apply(f, self.lam)) if self.adv else None
        return logits, dom, att
