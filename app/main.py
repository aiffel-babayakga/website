from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
from .services import IntegratedService
import os

app = FastAPI()

# ==============================================================================
# 🔐 CORS 미들웨어
# ==============================================================================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==============================================================================
# 📂 서비스 초기화
# ==============================================================================
base_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(base_dir)
data_dir = os.path.join(root_dir, "data")

print(f"📂 Data Directory: {data_dir}")

service = IntegratedService(
    fp_path=os.path.join(data_dir, "fp_smalltargets.pt"), 
    fr_path=os.path.join(data_dir, "fr_epoch6_20251227_052053.pt"), 
    vocab_path=os.path.join(data_dir, "fp_model_vocab.json"),
    gene_meta_path=os.path.join(data_dir, "gene_metadata.parquet")
)

# ==============================================================================
# API 엔드포인트
# ==============================================================================

class GeneInputPayload(BaseModel):
    genes: List[str]
    expressions: List[float]

@app.post("/predict/find_drug")
async def find_drug(payload: GeneInputPayload):
    if len(payload.genes) != len(payload.expressions):
        raise HTTPException(status_code=400, detail="유전자 개수와 발현량 불일치")
    if len(payload.genes) == 0:
        raise HTTPException(status_code=400, detail="유전자가 입력되지 않았습니다.")

    vector = service.predict_drug_from_genes(payload.genes, payload.expressions)
    
    if vector is None:
        raise HTTPException(status_code=400, detail="유효한 유전자가 없습니다.")

    return {"recommended_drug_vector": vector}


# ------------------------------------------------------------------------------
# 🛠️ [수정됨] 시뮬레이션 API
# ------------------------------------------------------------------------------
class SimulationPayload(BaseModel):
    smiles_embedding: List[float]
    genes: List[str]
    expressions: List[float]

@app.post("/predict/drug_response")
async def drug_response(payload: SimulationPayload):
    if not payload.smiles_embedding:
        raise HTTPException(status_code=400, detail="약물 벡터가 없습니다.")

    # 서비스 호출
    # result는 이제 {"top_genes": {...}, "pathways": {...}} 형태입니다.
    result = service.simulate_drug_response(
        payload.genes, 
        payload.expressions, 
        payload.smiles_embedding
    )
    
    if result is None:
        raise HTTPException(status_code=500, detail="FR 모델 로딩 실패 또는 예측 오류")

    # ✅ [중요 변경] result 자체를 반환해야 프론트엔드가 top_genes와 pathways를 모두 받습니다.
    # 기존: return {"top_genes": result} -> (X) 중복 포장됨
    # 변경: return result              -> (O)
    return result