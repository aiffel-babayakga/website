import torch
import torch.nn as nn
import numpy as np

# =========================================================
# 1. Encoder (With Organ Token, No Positional Embedding)
# =========================================================
class FPEncoderWithOrgan(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, num_layers, pad_id,
                 max_len: int, num_organs: int, organ_pos: int = 1, use_pos_emb: bool = False):
        super().__init__()
        self.token_emb  = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.value_proj = nn.Linear(1, d_model)

        # Positional Embedding 제거
        self.use_pos_emb = bool(use_pos_emb)
        if self.use_pos_emb:
            self.pos_emb = nn.Embedding(max_len, d_model)

        self.organ_emb = nn.Embedding(num_organs, d_model)
        self.organ_pos = int(organ_pos)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=4*d_model,
            dropout=0.1, batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

    def forward(self, input_ids, values, attention_mask, organ_id):
        B, L = input_ids.shape
        dev = input_ids.device

        x = self.token_emb(input_ids) + self.value_proj(values.unsqueeze(-1))

        if self.use_pos_emb:
            pos = torch.arange(L, device=dev).unsqueeze(0).expand(B, L)
            x = x + self.pos_emb(pos)

        if organ_id is not None:
            # Organ 토큰 위치에 임베딩 더하기
            if self.organ_pos < L:
                x[:, self.organ_pos, :] = x[:, self.organ_pos, :] + self.organ_emb(organ_id.to(dev)).to(x.dtype)

        key_padding_mask = (attention_mask == 0)
        h = self.encoder(x, src_key_padding_mask=key_padding_mask)
        return h[:, 0, :] # [CLS] 토큰의 출력 반환

# =========================================================
# 2. Main Model (FPModelTied_OrganCLIP)
# =========================================================
class FPModelTied_OrganCLIP(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, num_layers, pad_id, smiles_dim,
                 max_len: int, num_organs: int, n_special: int, tau_init: float = 0.10):
        super().__init__()
        self.n_special = int(n_special)

        self.encoder = FPEncoderWithOrgan(
            vocab_size=vocab_size,
            d_model=d_model,
            n_heads=n_heads,
            num_layers=num_layers,
            pad_id=pad_id,
            max_len=max_len,
            num_organs=num_organs,
            organ_pos=1,          # [CLS][ORGAN] 구조
            use_pos_emb=False,    
        )
        self.proj = nn.Linear(d_model, d_model)

        self.smiles_head = nn.Sequential(
            nn.Linear(d_model, 4*d_model),
            nn.BatchNorm1d(4*d_model),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(4*d_model, smiles_dim),
        )

        # CLIP logit scale
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1.0 / float(tau_init)))

    def gene_emb_subset(self):
        return self.encoder.token_emb.weight[self.n_special:, :]

    def get_tau(self):
        return (1.0 / self.logit_scale.exp()).clamp(0.01, 0.5)

    def forward(self, input_ids, values, attention_mask, organ_id, return_smiles=False):
        h_cls = self.encoder(input_ids, values, attention_mask, organ_id=organ_id)
        v_pred = self.proj(h_cls)
        z_pred = self.smiles_head(h_cls)
        if return_smiles:
            return v_pred, z_pred
        return v_pred

# =========================================================
# 3. FR Model Classes
# =========================================================
class Cell2SentenceEncoderFR(nn.Module):
    """
    prefix: [CLS][DRUG][CELL] + gene tokens
    """
    def __init__(self, vocab_size, d_model, n_heads, num_layers, max_len_with_prefix, smiles_dim, num_cell_lines, dropout=0.1, pad_id=0):
        super().__init__()
        self.d_model = d_model
        
        # pad_id를 인자로 받도록 처리
        self.token_emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.value_proj = nn.Sequential(
            nn.Linear(1, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.pos_emb = nn.Embedding(max_len_with_prefix, d_model)

        self.cell_line_emb = nn.Embedding(num_cell_lines, d_model)
        self.smiles_proj = nn.Linear(smiles_dim, d_model)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=4*d_model,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

    def forward(self, input_ids, values, attention_mask, cell_line_id, smiles_emb):
        B, L = input_ids.shape
        device_ = input_ids.device

        x = self.token_emb(input_ids) + self.value_proj(values.unsqueeze(-1))

        # Positional Embedding
        pos = torch.arange(L, device=device_).unsqueeze(0).expand(B, L)
        x = x + self.pos_emb(pos)

        # Inject drug / cell info (Position 1, 2)
        # smiles_emb: (B, 768) -> (B, d_model)
        x[:, 1, :] = x[:, 1, :] + self.smiles_proj(smiles_emb.to(device=device_, dtype=torch.float32)).to(x.dtype)
        x[:, 2, :] = x[:, 2, :] + self.cell_line_emb(cell_line_id.to(device=device_)).to(x.dtype)

        key_padding_mask = (attention_mask == 0)
        h = self.encoder(x, src_key_padding_mask=key_padding_mask)
        return h[:, 0, :]  # [CLS] token output


class FRModelExpression(nn.Module):
    def __init__(self, encoder, d_model, out_dim):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Linear(d_model, out_dim)

    def forward(self, input_ids, values, mask, cell_line_id, smiles_emb):
        h = self.encoder(input_ids, values, mask, cell_line_id, smiles_emb)
        return self.head(h)