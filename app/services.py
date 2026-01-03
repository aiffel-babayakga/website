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
    "MAX_SEQ_LEN": 256,       # MAX_SEQ_LEN = 256
    "PAD_ID": 0,              # local_token_to_id["[PAD]"]
    "CLS_ID": 1,              # local_token_to_id["[CLS]"]
    "ORGAN_TOK_ID": 2,        # local_token_to_id["[ORGAN]"]
    "DELTA_CLIP_ABS": 5.0,    # DELTA_CLIP_ABS = 5.0
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

# 데모용 Pathway 데이터베이스
GENE_PATHWAY_MAP = {
    "EGFR": "RTK Signaling", "KRAS": "MAPK Signaling", "BRAF": "MAPK Signaling",
    "PIK3CA": "PI3K-Akt Signaling", "PTEN": "PI3K-Akt Signaling", "AKT1": "PI3K-Akt Signaling",
    "TP53": "p53 Signaling / Cell Cycle", "CDKN2A": "Cell Cycle", "RB1": "Cell Cycle",
    "APC": "Wnt Signaling", "CTNNB1": "Wnt Signaling",
    "SMAD4": "TGF-beta Signaling", "TGFBR2": "TGF-beta Signaling",
    "BRCA1": "DNA Repair", "BRCA2": "DNA Repair",
    "MYC": "Cell Growth & Proliferation",
    "VEGFA": "Angiogenesis", "VIM": "EMT (Metastasis)", "FN1": "EMT (Metastasis)",
    "MALAT1": "Transcriptional Regulation", "NEAT1": "Nuclear Structure",
    "MT-RNR1": "Metabolic Process", "GAPDH": "Metabolic Process",
    "TNF": "Inflammation", "IL6": "Inflammation",
    "TRIO": "Cytoskeleton Organization", "ASPH": "Cell Motility",
    "HSP90B1": "Protein Folding (Stress Response)", "EXT1": "Heparan Sulfate Biosynthesis",
    "SPARC": "Extracellular Matrix Organization", "PDE4D": "cAMP Signaling",
    "MT-ND4": "Metabolic Process", "PDE10A": "cAMP Signaling", 
    "HSP90AA1": "Protein Folding (Stress Response)", "TMSB10": "Cytoskeleton Organization",
    "SERPINE1": "Extracellular Matrix Organization"
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
        
        # 2. FR용 Vocab 및 매핑 생성
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
        if not os.path.exists(meta_path):
            print(f"⚠️ Gene metadata not found at {meta_path}")
            return {}, {}
        
        df = pd.read_parquet(meta_path)
        
        # FR 입력용 Vocab
        special_tokens = ["[PAD]", "[CLS]", "[DRUG]", "[TARGET]", "[CELL]", "[MASK]"]
        local_token_to_id = {tok: i for i, tok in enumerate(special_tokens)}
        for ensg in df["ensembl_id"].astype(str):
            if ensg not in local_token_to_id:
                local_token_to_id[ensg] = len(local_token_to_id)
        
        # 결과 해석용 매핑
        tahoe_id_to_symbol = {}
        if 'token_id' in df.columns and 'gene_symbol' in df.columns:
            tahoe_id_to_symbol = dict(zip(df["token_id"].astype(int), df["gene_symbol"].astype(str)))
        
        return local_token_to_id, tahoe_id_to_symbol

    def _load_fp_model(self, path):
        model = FPModelTied_OrganCLIP(
            vocab_size=4188, d_model=256, n_heads=8, num_layers=4,
            pad_id=FP_CONFIG["PAD_ID"], smiles_dim=768, max_len=256+2, # +2 for CLS, ORGAN
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
        """
        학습 코드(TahoeFPParquetDataset)와 동일한 전처리 로직 적용
        """
        # 1. 유효한 유전자 필터링
        valid_inputs = []
        for name, val in zip(gene_names, gene_values):
            name_upper = str(name).upper()
            if name_upper in self.fp_vocab_map:
                # 학습 코드의 DELTA_CLIP_ABS = 5.0 적용
                # 프론트엔드 입력값은 이미 Delta(변화량)라고 가정
                val_clipped = np.clip(float(val), -FP_CONFIG["DELTA_CLIP_ABS"], FP_CONFIG["DELTA_CLIP_ABS"])
                valid_inputs.append((self.fp_vocab_map[name_upper], val_clipped))
        
        if not valid_inputs: return None

        # (1) 절대값(Magnitude) 기준 내림차순 정렬하여 상위 MAX_SEQ_LEN개 추출
        valid_inputs.sort(key=lambda x: abs(x[1]), reverse=True)
        valid_inputs = valid_inputs[:FP_CONFIG["MAX_SEQ_LEN"]]
        
        # (2) Token ID 기준 오름차순 정렬 (Stable Sort by Gene ID)
        valid_inputs.sort(key=lambda x: x[0])

        # 3. 입력 텐서 구성: [CLS] [ORGAN] [Gene1] [Gene2] ...
        input_ids = [FP_CONFIG["CLS_ID"], FP_CONFIG["ORGAN_TOK_ID"]]
        values = [0.0, 0.0] # CLS, ORGAN 자리는 0.0
        
        for tid, val in valid_inputs:
            input_ids.append(tid)
            values.append(val)
        
        # 4. 패딩 (Padding)
        pad_len = (FP_CONFIG["MAX_SEQ_LEN"] + 2) - len(input_ids)
        if pad_len > 0:
            input_ids.extend([FP_CONFIG["PAD_ID"]] * pad_len)
            values.extend([0.0] * pad_len)

        # 5. 텐서 변환 및 모델 입력
        inp = torch.tensor([input_ids], dtype=torch.long).to(self.device)
        val = torch.tensor([values], dtype=torch.float32).to(self.device)
        msk = (inp != FP_CONFIG["PAD_ID"]).long()
        org = torch.tensor([0], dtype=torch.long).to(self.device) # Organ ID는 0(UNK) 또는 임의값

        with torch.no_grad():
            _, z_pred = self.model_fp(inp, val, msk, organ_id=org, return_smiles=True)
        return z_pred.cpu().numpy().tolist()[0]

    def simulate_drug_response(self, gene_names, gene_values, drug_vector):
        if self.model_fr is None: return None

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

        with torch.no_grad():
            delta_pred = self.model_fr(inp_t, val_t, msk_t, cell_id, drug_emb)
        
        delta_np = delta_pred.cpu().numpy()[0] 

        # 결과 매핑 및 Pathway 분석
        result_genes = {}
        pathway_counts = {}

        # 변화량이 큰 상위 20개만 추출
        top_idx = np.argsort(np.abs(delta_np))[::-1][:20]
        
        for i in top_idx:
            val = float(delta_np[i])
            gene_name = f"Gene_{i}"
            
            if len(self.fr_gene_ids) > i:
                tahoe_id = int(self.fr_gene_ids[i])
                if tahoe_id in self.tahoe_id_to_symbol:
                    gene_name = self.tahoe_id_to_symbol[tahoe_id]
            
            result_genes[gene_name] = val

            # Pathway 분석
            if gene_name in GENE_PATHWAY_MAP:
                pathway = GENE_PATHWAY_MAP[gene_name]
                pathway_counts[pathway] = pathway_counts.get(pathway, 0) + abs(val)
            elif gene_name.startswith("MT-") or gene_name.startswith("RPL") or gene_name.startswith("RPS"):
                pathway_counts["Translation & Metabolism"] = pathway_counts.get("Translation & Metabolism", 0) + abs(val)
            else:
                pathway_counts["Unknown/Novel Pathway"] = pathway_counts.get("Unknown/Novel Pathway", 0) + abs(val)

        sorted_pathways = dict(sorted(pathway_counts.items(), key=lambda item: item[1], reverse=True))

        return {
            "top_genes": result_genes,      
            "pathways": sorted_pathways     
        }