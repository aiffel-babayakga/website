import torch
import numpy as np
import pandas as pd
import os
import json
from .models import FPModelTied_OrganCLIP, Cell2SentenceEncoderFR, FRModelExpression

# ------------------------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------------------------
FP_CONFIG = {
    "MAX_SEQ_LEN": 256,
    "PAD_ID": 0,
    "CLS_ID": 1,
    "ORGAN_TOK_ID": 2, 
}

FR_CONFIG = {
    "VOCAB_SIZE": 62716, 
    "D_MODEL": 256,
    "N_HEADS": 8,
    "NUM_LAYERS": 4,
    "MAX_LEN": 512,      
    "PREFIX_LEN": 3,     
    "SMILES_DIM": 768,
    "NUM_CELL_LINES": 50,
    "TOP_K": 1000,       
    "PAD_ID": 0,
    "CLS_ID": 1,
    "DRUG_TOK_ID": 2,
    "CELL_TOK_ID": 4
}

# ------------------------------------------------------------------------------
# SERVICE CLASS
# ------------------------------------------------------------------------------
class IntegratedService:
    def __init__(self, fp_path, fr_path, vocab_path, gene_meta_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Running on device: {self.device}")

        # 1. FP용 Vocab 로드
        self.fp_vocab_map = self._load_json_vocab(vocab_path)
        
        # 2. FR용 Vocab 및 매핑 생성 (중요!)
        # tahoe_id_to_symbol: 모델의 숫자 출력을 유전자 이름으로 바꾸는 사전
        print("Build FR Vocab & Mapping...")
        self.fr_vocab_map, self.tahoe_id_to_symbol = self._build_fr_vocab(gene_meta_path)

        # 3. 모델 로드
        self.model_fp = self._load_fp_model(fp_path)
        self.model_fr, self.fr_gene_ids = self._load_fr_model(fr_path)

    def _load_json_vocab(self, path):
        if not os.path.exists(path): return {}
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return data.get('vocab_map', {})

    def _build_fr_vocab(self, meta_path):
        """메타데이터에서 Token ID -> Gene Symbol 매핑 정보를 생성합니다."""
        if not os.path.exists(meta_path):
            print(f"⚠️ Gene metadata not found at {meta_path}")
            return {}, {}
        
        df = pd.read_parquet(meta_path)
        
        # 1. FR 입력용 Vocab (Ensembl ID -> Index)
        special_tokens = ["[PAD]", "[CLS]", "[DRUG]", "[TARGET]", "[CELL]", "[MASK]"]
        local_token_to_id = {tok: i for i, tok in enumerate(special_tokens)}
        for ensg in df["ensembl_id"].astype(str):
            if ensg not in local_token_to_id:
                local_token_to_id[ensg] = len(local_token_to_id)
        
        # 2. 결과 해석용 매핑 (Tahoe Token ID -> Gene Symbol)
        # 데이터프레임의 'token_id' 컬럼과 'gene_symbol' 컬럼을 매핑
        tahoe_id_to_symbol = {}
        if 'token_id' in df.columns and 'gene_symbol' in df.columns:
            tahoe_id_to_symbol = dict(zip(df["token_id"].astype(int), df["gene_symbol"].astype(str)))
        
        return local_token_to_id, tahoe_id_to_symbol

    def _load_fp_model(self, path):
        model = FPModelTied_OrganCLIP(
            vocab_size=4188, d_model=256, n_heads=8, num_layers=4,
            pad_id=FP_CONFIG["PAD_ID"], smiles_dim=768, max_len=256+2,
            num_organs=16, n_special=4, tau_init=0.1
        )
        if os.path.exists(path):
            try:
                ckpt = torch.load(path, map_location=self.device)
                state_dict = ckpt['model_state'] if 'model_state' in ckpt else ckpt
                model.load_state_dict(state_dict, strict=True)
                print("✅ FP Model Loaded")
            except Exception as e: print(f"❌ FP Load Error: {e}")
        model.to(self.device)
        model.eval()
        return model

    def _load_fr_model(self, path):
        encoder = Cell2SentenceEncoderFR(
            vocab_size=FR_CONFIG["VOCAB_SIZE"], d_model=FR_CONFIG["D_MODEL"],
            n_heads=FR_CONFIG["N_HEADS"], num_layers=FR_CONFIG["NUM_LAYERS"],
            max_len_with_prefix=FR_CONFIG["PREFIX_LEN"] + FR_CONFIG["MAX_LEN"],
            smiles_dim=FR_CONFIG["SMILES_DIM"], num_cell_lines=FR_CONFIG["NUM_CELL_LINES"],
            pad_id=FR_CONFIG["PAD_ID"]
        )
        model = FRModelExpression(encoder, FR_CONFIG["D_MODEL"], out_dim=FR_CONFIG["TOP_K"])
        
        sorted_gene_ids = []
        if os.path.exists(path):
            try:
                ckpt = torch.load(path, map_location=self.device)
                if 'model_state' in ckpt: model.load_state_dict(ckpt['model_state'], strict=False)
                
                # 모델이 예측하는 유전자들의 Token ID 순서 가져오기
                if 'extra' in ckpt and 'sorted_gene_token_ids' in ckpt['extra']:
                    sorted_gene_ids = ckpt['extra']['sorted_gene_token_ids']
                    print(f"✅ FR Model Loaded (Targets: {len(sorted_gene_ids)} genes)")
            except Exception as e: print(f"❌ FR Load Error: {e}")
        
        model.to(self.device)
        model.eval()
        return model, sorted_gene_ids

    # --------------------------------------------------------------------------
    # PREDICTION FUNCTIONS
    # --------------------------------------------------------------------------
    def predict_drug_from_genes(self, gene_names, gene_values):
        valid_inputs = []
        for name, val in zip(gene_names, gene_values):
            name_upper = str(name).upper()
            if name_upper in self.fp_vocab_map:
                valid_inputs.append((self.fp_vocab_map[name_upper], float(val)))
        
        if not valid_inputs: return None
        valid_inputs.sort(key=lambda x: abs(x[1]), reverse=True)
        valid_inputs = valid_inputs[:FP_CONFIG["MAX_SEQ_LEN"]]
        valid_inputs.sort(key=lambda x: x[0])

        input_ids = [FP_CONFIG["CLS_ID"], FP_CONFIG["ORGAN_TOK_ID"]]
        values = [0.0, 0.0]
        for tid, val in valid_inputs:
            input_ids.append(tid)
            values.append(val)
        
        pad_len = (FP_CONFIG["MAX_SEQ_LEN"] + 2) - len(input_ids)
        if pad_len > 0:
            input_ids.extend([FP_CONFIG["PAD_ID"]] * pad_len)
            values.extend([0.0] * pad_len)

        inp = torch.tensor([input_ids], dtype=torch.long).to(self.device)
        val = torch.tensor([values], dtype=torch.float32).to(self.device)
        msk = (inp != FP_CONFIG["PAD_ID"]).long()
        org = torch.tensor([0], dtype=torch.long).to(self.device)

        with torch.no_grad():
            _, z_pred = self.model_fp(inp, val, msk, organ_id=org, return_smiles=True)
        return z_pred.cpu().numpy().tolist()[0]

    def simulate_drug_response(self, gene_names, gene_values, drug_vector):
        if self.model_fr is None: return None

        # 1. 입력 생성 (현재는 gene_names 매핑 로직 단순화: 0으로 채움)
        input_ids = [FR_CONFIG["CLS_ID"], FR_CONFIG["DRUG_TOK_ID"], FR_CONFIG["CELL_TOK_ID"]]
        values = [0.0, 0.0, 0.0]
        mask = [1, 1, 1]

        pad_len = (FR_CONFIG["PREFIX_LEN"] + FR_CONFIG["MAX_LEN"]) - len(input_ids)
        if pad_len > 0:
            input_ids.extend([FR_CONFIG["PAD_ID"]] * pad_len)
            values.extend([0.0] * pad_len)
            mask.extend([0] * pad_len)

        inp_t = torch.tensor([input_ids], dtype=torch.long).to(self.device)
        val_t = torch.tensor([values], dtype=torch.float32).to(self.device)
        msk_t = torch.tensor([mask], dtype=torch.long).to(self.device)
        cell_id = torch.tensor([0], dtype=torch.long).to(self.device)
        drug_emb = torch.tensor([drug_vector], dtype=torch.float32).to(self.device)

        # 2. 예측
        with torch.no_grad():
            delta_pred = self.model_fr(inp_t, val_t, msk_t, cell_id, drug_emb)
        
        delta_np = delta_pred.cpu().numpy()[0] 

        # 3. [수정됨] 결과 매핑 (Target_Gene_X -> 실제 유전자 이름)
        result = {}
        # 변화량이 큰 상위 15개만 추출
        top_idx = np.argsort(np.abs(delta_np))[::-1][:15]
        
        for i in top_idx:
            val = float(delta_np[i])
            gene_name = f"Gene_{i}" # 기본값
            
            # 저장된 유전자 순서 정보(sorted_gene_ids)가 있으면 이름 찾기
            if len(self.fr_gene_ids) > i:
                tahoe_id = int(self.fr_gene_ids[i]) # Tahoe Token ID
                if tahoe_id in self.tahoe_id_to_symbol:
                    gene_name = self.tahoe_id_to_symbol[tahoe_id] # 실제 이름 (예: EGFR)
            
            result[gene_name] = val

        return result