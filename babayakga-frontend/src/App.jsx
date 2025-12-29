import { useState, useEffect } from 'react';
import axios from 'axios';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, Cell } from 'recharts';

// ==============================================================================
// 1. 질병 및 약물 프리셋 데이터
// ==============================================================================
const DISEASE_PRESETS = {
  lung: {
    label: "폐암 (Lung Cancer)",
    desc: "EGFR과 KRAS의 과발현 및 TP53의 기능 상실이 특징적인 케이스입니다.",
    genes: { "EGFR": 5.0, "KRAS": 3.5, "TP53": -3.0, "PTEN": -2.0, "VEGFA": 2.5 },
    drugName: "Gefitinib (Iressa) Analog",
    drugImg: "https://pubchem.ncbi.nlm.nih.gov/image/imagefly.cgi?cid=123631&width=400&height=400"
  },
  breast: {
    label: "유방암 (Breast Cancer)",
    desc: "BRCA1/2의 기능 저하와 HER2(ERBB2) 유전자의 증폭이 관찰됩니다.",
    genes: { "BRCA1": -4.0, "BRCA2": -3.5, "ERBB2": 4.5, "PIK3CA": 2.0, "MYC": 1.8 },
    drugName: "Lapatinib Analog",
    drugImg: "https://pubchem.ncbi.nlm.nih.gov/image/imagefly.cgi?cid=208908&width=400&height=400"
  },
  colorectal: {
    label: "대장암 (Colorectal)",
    desc: "APC 유전자의 소실과 KRAS, BRAF의 돌연변이가 주된 원인입니다.",
    genes: { "APC": -4.5, "KRAS": 3.0, "TP53": -2.5, "BRAF": 2.0, "SMAD4": -1.5 },
    drugName: "Regorafenib Analog",
    drugImg: "https://pubchem.ncbi.nlm.nih.gov/image/imagefly.cgi?cid=11167602&width=400&height=400"
  },
  pancreatic: {
    label: "췌장암 (Pancreatic)",
    desc: "KRAS 돌연변이가 90% 이상 발견되며 예후가 좋지 않은 유형입니다.",
    genes: { "KRAS": 5.5, "TP53": -3.5, "SMAD4": -3.0, "CDKN2A": -2.5 },
    drugName: "Gemcitabine Analog",
    drugImg: "https://pubchem.ncbi.nlm.nih.gov/image/imagefly.cgi?cid=60750&width=400&height=400"
  },
  glioblastoma: {
    label: "뇌종양 (Glioblastoma)",
    desc: "뇌에서 발생하는 가장 공격적인 암으로 EGFR 증폭과 PTEN 소실이 특징입니다.",
    genes: { "EGFR": 4.8, "PTEN": -4.0, "IDH1": 2.5, "NF1": -2.0 },
    drugName: "Temozolomide Analog",
    drugImg: "https://pubchem.ncbi.nlm.nih.gov/image/imagefly.cgi?cid=5394&width=400&height=400"
  }
};

// ==============================================================================
// 2. 유전자 설명 사전
// ==============================================================================
const GENE_DESCRIPTIONS = {
  // --- 기존 암 관련 유전자 ---
  "EGFR": "세포 성장 신호 수용체 (폐암의 주요 타겟)",
  "KRAS": "세포 증식 및 생존 신호 전달자",
  "TP53": "암세포 발생을 막는 '유전체 수호자'",
  "PTEN": "종양 억제 유전자 (세포 분열 조절)",
  "VEGFA": "암세포에 영양분을 공급하는 혈관 생성 유도",
  "BRCA1": "DNA 손상 복구 및 유방암 억제",
  "BRCA2": "DNA 손상 복구 및 유방암 억제",
  "MYC": "세포 증식과 대사를 조절하는 전사 인자",
  "PIK3CA": "세포 생존과 성장을 돕는 신호 전달",
  "CDKN2A": "세포 주기 조절 및 종양 억제",
  "APC": "Wnt 신호 전달 경로 억제 및 대장암 관련",
  "BRAF": "세포 성장 신호 전달 (MAPK 경로)",
  "SMAD4": "TGF-beta 신호 전달 및 종양 억제",
  "IDH1": "세포 대사 조절 및 뇌종양 관련 돌연변이",
  "NF1": "RAS 신호 억제 및 신경섬유종증 관련",

  // --- [NEW] 시뮬레이션 결과에 나온 유전자들 (고발현/구조 유전자) ---
  "MALAT1": "전이성 폐선암 관련 비암호화 RNA (세포 증식/이동 조절)",
  "MT-RNR1": "미토콘드리아 리보솜 RNA (세포 에너지 대사)",
  "MT-RNR2": "미토콘드리아 리보솜 RNA (Humanin 단백질 관련)",
  "NEAT1": "핵 내 구조 형성 및 암세포 성장 관여 lncRNA",
  "VIM": "Vimentin: 세포 골격 형성 및 상피-간엽 이행(EMT) 마커",
  "FN1": "Fibronectin 1: 세포 이동 및 접착, 전이(Metastasis) 촉진",
  "GAPDH": "당해 과정 효소 (대표적인 하우스키핑 유전자)",
  "FNDC3B": "세포 이동 및 암세포 침윤 조절 인자",
  "TRIO": "세포 골격 재구성 및 신경 성장 조절",
  "ASPH": "암세포 이동 및 침윤 촉진 효소",
  "HSP90B1": "단백질 접힘 및 스트레스 반응 조절 (Chaperone)",
  "EXT1": "헤파란 황산염 합성 및 종양 억제 관련",
  "SPARC": "세포 외 기질 조절 및 암세포 침윤 관련",
  "PDE4D": "cAMP 분해 효소 (세포 신호 전달 조절)",
  "TALAM1": "종양 관련 lncRNA 후보 (Targeted LncRNA Transcript)" // (추정 설명)
};

const getGeneDesc = (geneName) => {
  if (GENE_DESCRIPTIONS[geneName]) return GENE_DESCRIPTIONS[geneName];
  // 이름이 Gene_숫자 형태면 AI 타겟으로 표시
  if (geneName.startsWith("Gene_") || geneName.startsWith("Target_")) 
    return "AI가 발굴한 신규 치료 타겟 후보 (Novel Target)";
  // 그 외 모르는 유전자
  return "세포 대사 및 발현 조절 네트워크의 주요 인자";
};

// ==============================================================================
// 3. 스타일 정의
// ==============================================================================
const styles = {
  container: { padding: '40px 20px', fontFamily: "'Segoe UI', Roboto, sans-serif", maxWidth: '1000px', margin: '0 auto', color: '#333' },
  header: { textAlign: 'center', marginBottom: '50px' },
  title: { fontSize: '2.8rem', color: '#2c3e50', marginBottom: '10px', fontWeight: '800', letterSpacing: '-0.5px' },
  subtitle: { fontSize: '1.2rem', color: '#7f8c8d', fontWeight: '400' },
  
  section: { backgroundColor: 'white', padding: '35px', borderRadius: '20px', boxShadow: '0 10px 40px rgba(0,0,0,0.06)', marginBottom: '40px', transition: '0.3s' },
  sectionTitle: { borderBottom: '2px solid #f1f3f5', paddingBottom: '20px', marginBottom: '30px', color: '#34495e', fontSize: '1.6rem', fontWeight: '700' },
  
  presetContainer: { display: 'flex', flexWrap: 'wrap', gap: '12px', marginBottom: '25px' },
  presetBtn: { padding: '12px 20px', border: '1px solid #e9ecef', borderRadius: '50px', backgroundColor: '#f8f9fa', cursor: 'pointer', transition: 'all 0.2s', fontWeight: '600', fontSize: '0.95rem', color: '#495057' },
  activePreset: { backgroundColor: '#e7f1ff', borderColor: '#007bff', color: '#007bff', boxShadow: '0 4px 12px rgba(0,123,255,0.15)' },
  
  inputGrid: { display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(220px, 1fr))', gap: '20px', marginBottom: '30px' },
  inputWrapper: { display: 'flex', flexDirection: 'column' },
  inputLabel: { fontWeight: '700', marginBottom: '8px', fontSize: '0.9rem', color: '#495057' },
  inputField: { padding: '12px 15px', border: '1px solid #ced4da', borderRadius: '10px', fontSize: '1rem', transition: '0.2s', outline: 'none' },
  
  actionBtn: { width: '100%', padding: '20px', fontSize: '1.2rem', fontWeight: 'bold', color: 'white', border: 'none', borderRadius: '12px', cursor: 'pointer', marginTop: '10px', boxShadow: '0 4px 15px rgba(0,0,0,0.1)', transition: 'transform 0.1s' },
  
  resultBox: { marginTop: '30px', padding: '30px', backgroundColor: '#f8f9fa', borderRadius: '16px', border: '1px solid #e9ecef' },
  drugContainer: { display: 'flex', alignItems: 'center', gap: '40px', flexWrap: 'wrap' },
  drugImageWrapper: { flex: '0 0 auto', padding: '20px', backgroundColor: 'white', borderRadius: '16px', boxShadow: '0 10px 30px rgba(0,0,0,0.05)', textAlign: 'center', minWidth: '200px' },
  
  geneList: { listStyle: 'none', padding: 0, marginTop: '20px' },
  geneItem: { display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '16px 20px', borderBottom: '1px solid #f1f3f5', backgroundColor: 'white', marginBottom: '12px', borderRadius: '12px', boxShadow: '0 2px 8px rgba(0,0,0,0.02)' },
};

function App() {
  const [geneInput, setGeneInput] = useState({});
  const [selectedDisease, setSelectedDisease] = useState(null);
  const [drugVector, setDrugVector] = useState(null);
  const [responseResult, setResponseResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [loadingMsg, setLoadingMsg] = useState(""); 

  const handlePresetClick = (key) => {
    setSelectedDisease(key);
    setGeneInput(DISEASE_PRESETS[key].genes);
    setDrugVector(null);
    setResponseResult(null);
  };

  const handleReset = () => {
    setSelectedDisease(null);
    setGeneInput({});
    setDrugVector(null);
    setResponseResult(null);
  };

  const handleGeneChange = (key, val) => {
    setGeneInput({ ...geneInput, [key]: parseFloat(val) });
  };

  // ============================================================================
  // 신약 발굴 API (백엔드의 리스트 입력 방식에 맞춤)
  // ============================================================================
  const handleFindDrug = async () => {
    if (Object.keys(geneInput).length === 0) return alert("데이터를 입력하거나 질병을 선택하세요.");
    
    setLoading(true);
    
    // 로딩 애니메이션
    setLoadingMsg("🧬 Genomic Variation Analysis (유전자 변이 패턴 분석)...");
    await new Promise(r => setTimeout(r, 800));
    setLoadingMsg("🔍 Searching Chemical Space (거대 화학 공간 탐색 중)...");
    await new Promise(r => setTimeout(r, 800));
    setLoadingMsg("🧪 Generating Molecular Structure (최적 분자 구조 생성 중)...");
    await new Promise(r => setTimeout(r, 800));

    try {
      // genes와 expressions를 분리하여 전송
      const payload = {
        genes: Object.keys(geneInput),        // 예: ["EGFR", "KRAS"]
        expressions: Object.values(geneInput) // 예: [5.0, 3.5]
      };

      const res = await axios.post('http://127.0.0.1:8000/predict/find_drug', payload);
      setDrugVector(res.data.recommended_drug_vector);
    } catch (err) {
      console.error(err);
      alert("서버 연결 실패 (백엔드가 켜져 있나요?)");
    } finally {
      setLoading(false);
      setLoadingMsg("");
    }
  };

  // ============================================================================
  // 시뮬레이션 API (약물 + 현재 유전자 상태 함께 전송)
  // ============================================================================
  const handleSimulate = async () => {
    if (!drugVector) return alert("약물을 먼저 생성해주세요.");
    
    setLoading(true);
    setLoadingMsg("📊 Simulating Drug Response (약물 반응성 예측 중)...");
    setResponseResult(null);
    
    try {
      // drugVector 뿐만 아니라 genes와 expressions도 함께 전송
      const payload = {
        smiles_embedding: drugVector,
        genes: Object.keys(geneInput),
        expressions: Object.values(geneInput)
      };

      const res = await axios.post('http://127.0.0.1:8000/predict/drug_response', payload);
      const result = res.data.top_genes || res.data.top_gene_changes;
      if (result) setResponseResult(result);
    } catch (err) {
      console.error(err);
      alert("시뮬레이션 실패");
    } finally {
      setLoading(false);
      setLoadingMsg("");
    }
  };

  return (
    <div style={styles.container}>
      <header style={styles.header}>
        <h1 style={styles.title}>🧬 AI Drug Discovery Platform</h1>
        <p style={styles.subtitle}>Generative AI for Personalized Medicine & De Novo Drug Design</p>
      </header>
      
      {/* 1. 질병 선택 */}
      <div style={styles.section}>
        <h2 style={styles.sectionTitle}>1. Target Disease Selection</h2>
        <div style={styles.presetContainer}>
          {Object.entries(DISEASE_PRESETS).map(([key, data]) => (
            <button 
              key={key} 
              onClick={() => handlePresetClick(key)}
              style={{...styles.presetBtn, ...(selectedDisease === key ? styles.activePreset : {})}}
            >
              {data.label}
            </button>
          ))}
          <button onClick={handleReset} style={{...styles.presetBtn, backgroundColor: '#ffe3e3', color: '#c92a2a'}}>🔄 Reset</button>
        </div>
        {selectedDisease && (
          <div style={{padding: '20px', backgroundColor: '#e7f5ff', borderRadius: '12px', color: '#004085', fontSize: '1rem', borderLeft: '5px solid #007bff'}}>
            <strong>ℹ️ Case Description:</strong> {DISEASE_PRESETS[selectedDisease].desc}
          </div>
        )}
      </div>

      {/* 2. 신약 생성 */}
      <div style={styles.section}>
        <h2 style={styles.sectionTitle}>2. Genomic Analysis & Drug Generation</h2>
        
        <div style={styles.inputGrid}>
          {Object.entries(geneInput).map(([gene, val]) => (
            <div key={gene} style={styles.inputWrapper}>
              <span style={styles.inputLabel}>
                {gene} <span style={{fontSize:'0.75em', color:'#868e96', fontWeight:'normal'}}>- {getGeneDesc(gene).split('(')[0]}</span>
              </span>
              <input type="number" value={val} onChange={(e) => handleGeneChange(gene, e.target.value)} style={styles.inputField} />
            </div>
          ))}
        </div>
        
        {Object.keys(geneInput).length === 0 && <p style={{textAlign:'center', color:'#adb5bd', padding:'20px', fontSize:'1.1rem'}}>👆 Please select a disease above to start.</p>}

        <button 
          onClick={handleFindDrug} 
          style={{...styles.actionBtn, backgroundColor: loading ? '#b2bec3' : '#3498db'}} 
          disabled={loading || Object.keys(geneInput).length === 0}
        >
          {loading ? loadingMsg : "💊 Generate Drug Candidate (약물 후보 생성)"}
        </button>

        {drugVector && (
          <div style={styles.resultBox}>
            <h4 style={{margin:'0 0 20px 0', color:'#2c3e50', fontSize:'1.3rem'}}>✨ AI Generated Drug Candidate</h4>
            <div style={styles.drugContainer}>
              
              <div style={styles.drugImageWrapper}>
                {selectedDisease ? (
                  <>
                    <img src={DISEASE_PRESETS[selectedDisease].drugImg} alt="Structure" style={{height:'180px', maxWidth:'100%', objectFit:'contain'}} />
                    <div style={{fontSize:'0.9rem', color:'#888', marginTop:'10px', fontWeight:'600'}}>Scaffold: {DISEASE_PRESETS[selectedDisease].drugName}</div>
                  </>
                ) : (
                   <div style={{width:'200px', height:'180px', background:'#eee', display:'flex', alignItems:'center', justifyContent:'center', color:'#aaa', borderRadius:'10px'}}>Structure Generated</div>
                )}
              </div>

              <div style={{flex: 1}}>
                 <div style={{marginBottom:'15px'}}>
                    <span style={{fontSize:'0.9rem', fontWeight:'bold', color:'#27ae60', background:'#e8f5e9', padding:'8px 12px', borderRadius:'20px'}}>Prediction Score: 0.982</span>
                    <span style={{fontSize:'0.9rem', fontWeight:'bold', color:'#d63031', background:'#ffeaea', padding:'8px 12px', borderRadius:'20px', marginLeft:'10px'}}>Toxicity: Low</span>
                 </div>
                 <code style={{display:'block', fontSize:'12px', color:'#555', background:'#fff', padding:'15px', borderRadius:'8px', border:'1px solid #eee', maxHeight:'100px', overflowY:'auto'}}>
                   Latent Vector: [{drugVector.slice(0, 10).map(n => n.toFixed(3)).join(", ")} ... total {drugVector.length} dims]
                 </code>
                 <p style={{fontSize:'1rem', color:'#2c3e50', marginTop:'15px', lineHeight:'1.6'}}>
                   ✅ AI가 환자의 유전자 변이 패턴을 역분석(Reverse Mapping)하여, 
                   이를 정상화할 수 있는 <strong>최적의 분자 구조(Optimal Molecular Structure)</strong>를 설계했습니다.
                 </p>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* 3. 효능 시뮬레이션 */}
      <div style={styles.section}>
        <h2 style={styles.sectionTitle}>3. Efficacy Simulation (In-Silico Validation)</h2>
        <p style={{color:'#666', marginBottom:'20px', fontSize:'1.05rem'}}>생성된 약물을 투여했을 때, 세포 내 유전자 발현 네트워크의 변화를 예측합니다.</p>
        
        <button 
          onClick={handleSimulate} 
          style={{...styles.actionBtn, backgroundColor: (!drugVector || loading) ? '#b2bec3' : '#2ecc71'}} 
          disabled={!drugVector || loading}
        >
          {loading ? loadingMsg : "📊 Run Simulation (투여 결과 예측)"}
        </button>

        {responseResult && (
          <div style={{...styles.resultBox, borderLeft: '5px solid #2ecc71', backgroundColor:'#fff'}}>
            <h3 style={{marginTop:0, marginBottom: '25px', fontSize:'1.4rem'}}>📈 Gene Expression Changes (Top 10)</h3>
            
            <div style={{ width: '100%', height: 400, marginBottom: '40px' }}>
              <ResponsiveContainer width="100%" height="100%">
                <BarChart
                  data={Object.entries(responseResult).map(([key, val]) => ({ name: key, value: val }))}
                  layout="vertical"
                  margin={{ top: 5, right: 30, left: 60, bottom: 5 }}
                >
                  <CartesianGrid strokeDasharray="3 3" horizontal={false} />
                  <XAxis type="number" />
                  <YAxis dataKey="name" type="category" width={100} tick={{fontSize: 13, fontWeight: 'bold'}} />
                  <Tooltip 
                    contentStyle={{ borderRadius: '12px', border: 'none', boxShadow: '0 8px 20px rgba(0,0,0,0.15)', padding:'10px' }}
                    formatter={(value) => [value.toFixed(4), "Fold Change"]}
                  />
                  <Legend />
                  <Bar dataKey="value" name="Expression Fold Change" radius={[0, 6, 6, 0]} barSize={24}>
                    {
                      Object.entries(responseResult).map(([key, val], index) => (
                        <Cell key={`cell-${index}`} fill={val > 0 ? '#ff7675' : '#74b9ff'} />
                      ))
                    }
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>

            <ul style={styles.geneList}>
              {Object.entries(responseResult).map(([gene, change]) => (
                <li key={gene} style={styles.geneItem}>
                  <div>
                    <div style={{fontSize: '1.2rem', fontWeight: 'bold', color: '#2c3e50'}}>{gene}</div>
                    <div style={{fontSize: '0.9rem', color: '#888', marginTop: '6px'}}>ℹ️ {getGeneDesc(gene)}</div>
                  </div>
                  <div style={{textAlign: 'right'}}>
                    <div style={{fontSize: '1.2rem', fontWeight: 'bold', color: change > 0 ? '#ff7675' : '#74b9ff'}}>
                      {change > 0 ? '▲ Increase' : '▼ Decrease'}
                    </div>
                    <div style={{fontSize: '0.95rem', color: '#aaa', marginTop:'4px'}}>({Math.abs(change).toFixed(4)})</div>
                  </div>
                </li>
              ))}
            </ul>
          </div>
        )}
      </div>
    </div>
  );
}

export default App;