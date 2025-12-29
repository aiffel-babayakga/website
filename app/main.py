from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
from .services import IntegratedService
import os

app = FastAPI()

# ==============================================================================
# 🔐 CORS 미들웨어 (보안 설정)
# ==============================================================================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==============================================================================
# 📂 경로 설정 및 서비스 초기화
# ==============================================================================
base_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(base_dir)
data_dir = os.path.join(root_dir, "data")

print(f"📂 Data Directory: {data_dir}")

# gene_meta_path 인자가 추가되었습니다!
service = IntegratedService(
    fp_path=os.path.join(data_dir, "fp_smalltargets.pt"), 
    fr_path=os.path.join(data_dir, "fr_epoch6_20251227_052053.pt"), 
    vocab_path=os.path.join(data_dir, "fp_model_vocab.json"),
    gene_meta_path=os.path.join(data_dir, "gene_metadata.parquet")
)

# ==============================================================================
# API 엔드포인트
# ==============================================================================

# 1. 신약 생성 요청 모델
class GeneInputPayload(BaseModel):
    genes: List[str]
    expressions: List[float]

@app.post("/predict/find_drug")
async def find_drug(payload: GeneInputPayload):
    # 데이터 검증
    if len(payload.genes) != len(payload.expressions):
        raise HTTPException(status_code=400, detail="유전자 개수와 발현량 개수가 다릅니다.")
    
    if len(payload.genes) == 0:
        raise HTTPException(status_code=400, detail="유전자가 입력되지 않았습니다.")

    # 서비스 호출
    vector = service.predict_drug_from_genes(payload.genes, payload.expressions)
    
    if vector is None:
        raise HTTPException(status_code=400, detail="입력된 유전자 중 모델이 아는 유전자가 하나도 없습니다.")

    return {"recommended_drug_vector": vector}


# 시뮬레이션 요청 모델 추가
class SimulationPayload(BaseModel):
    smiles_embedding: List[float] # 약물 벡터
    genes: List[str]              # 현재 유전자 이름
    expressions: List[float]      # 현재 유전자 발현량

# 시뮬레이션 엔드포인트 구현
@app.post("/predict/drug_response")
async def drug_response(payload: SimulationPayload):
    # 약물 벡터 확인
    if not payload.smiles_embedding:
        raise HTTPException(status_code=400, detail="약물 벡터가 없습니다.")

    # 서비스 호출 (인자 3개를 넘겨줍니다)
    result = service.simulate_drug_response(
        payload.genes, 
        payload.expressions, 
        payload.smiles_embedding
    )
    
    if result is None:
        raise HTTPException(status_code=500, detail="FR 모델 로딩 실패 또는 예측 오류")

    # 결과 반환
    return {"top_genes": result}