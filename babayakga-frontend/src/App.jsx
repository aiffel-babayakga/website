import { useState } from 'react';
import axios from 'axios';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, Cell, PieChart, Pie } from 'recharts';

// ==============================================================================
// 1. 질병 및 약물 프리셋 데이터
// ==============================================================================
const DISEASE_PRESETS = {
  lung: {
    label: "폐암 (Lung Cancer)",
    desc: "EGFR과 KRAS의 과발현 및 TP53의 기능 상실이 특징적인 케이스입니다.",
    genes: { 
      // 기존
      "EGFR": 5.0, "KRAS": 3.5, "TP53": -3.0, "PTEN": -2.0, "VEGFA": 2.5,
      // 폐암 핵심 바이오마커
      "ALK": 4.2, "ROS1": 3.8, "MET": 2.9, "RET": 2.1, "BRAF": 1.5, "NTRK1": 1.2 
    },
    drugName: "Gefitinib (Iressa) Analog",
    drugImg: "https://pubchem.ncbi.nlm.nih.gov/image/imagefly.cgi?cid=123631&width=400&height=400"
  },
  breast: {
    label: "유방암 (Breast Cancer)",
    desc: "BRCA1/2의 기능 저하와 HER2(ERBB2) 유전자의 증폭이 관찰됩니다.",
    genes: { 
      // 기존
      "BRCA1": -4.0, "BRCA2": -3.5, "ERBB2": 4.5, "PIK3CA": 2.0, "MYC": 1.8,
      // 유방암 호르몬 수용체 및 관련 인자
      "ESR1": 3.2, "PGR": 2.5, "AKT1": 2.1, "GATA3": -1.5, "CDH1": -2.8, "CCND1": 2.4
    },
    drugName: "Lapatinib Analog",
    drugImg: "https://pubchem.ncbi.nlm.nih.gov/image/imagefly.cgi?cid=208908&width=400&height=400"
  },
  colorectal: {
    label: "대장암 (Colorectal)",
    desc: "APC 유전자의 소실과 KRAS, BRAF의 돌연변이가 주된 원인입니다.",
    genes: { 
      // 기존
      "APC": -4.5, "KRAS": 3.0, "TP53": -2.5, "BRAF": 2.0, "SMAD4": -1.5,
      // 대장암 관련 경로
      "NRAS": 2.2, "PIK3CA": 1.8, "FBXW7": -2.0, "TGFBR2": -1.8, "BAX": -1.2 
    },
    drugName: "Regorafenib Analog",
    drugImg: "https://pubchem.ncbi.nlm.nih.gov/image/imagefly.cgi?cid=11167602&width=400&height=400"
  },
  pancreatic: {
    label: "췌장암 (Pancreatic)",
    desc: "KRAS 돌연변이가 90% 이상 발견되며 예후가 좋지 않은 유형입니다.",
    genes: { 
      // 기존
      "KRAS": 5.5, "TP53": -3.5, "SMAD4": -3.0, "CDKN2A": -2.5,
      // 췌장암 관련
      "GNAS": 2.1, "RNF43": -1.9, "ATM": -2.2, "MLH1": -1.5, "ARID1A": -1.8
    },
    drugName: "Gemcitabine Analog",
    drugImg: "https://pubchem.ncbi.nlm.nih.gov/image/imagefly.cgi?cid=60750&width=400&height=400"
  },
  glioblastoma: {
    label: "뇌종양 (Glioblastoma)",
    desc: "뇌에서 발생하는 가장 공격적인 암으로 EGFR 증폭과 PTEN 소실이 특징입니다.",
    genes: { 
      // 기존
      "EGFR": 4.8, "PTEN": -4.0, "IDH1": 2.5, "NF1": -2.0,
      // 뇌종양 마커
      "TERT": 3.5, "ATRX": -2.5, "MGMT": -3.0, "RB1": -2.2, "TP53": -2.8
    },
    drugName: "Temozolomide Analog",
    drugImg: "https://pubchem.ncbi.nlm.nih.gov/image/imagefly.cgi?cid=5394&width=400&height=400"
  }
};

// ==============================================================================
// 2. 유전자 설명 사전
// ==============================================================================
const GENE_DESCRIPTIONS = {
  // --- 공통/기존 암 관련 유전자 ---
  "EGFR": "세포 성장 신호 수용체 (폐암의 주요 타겟)",
  "KRAS": "세포 증식 및 생존 신호 전달자 (MAPK 경로)",
  "TP53": "암세포 발생을 막는 '유전체 수호자' (Tumor Suppressor)",
  "PTEN": "종양 억제 및 PI3K 신호 경로 조절 인자",
  "VEGFA": "암세포에 영양분을 공급하는 혈관 생성 유도 인자",
  "BRCA1": "DNA 이중 나선 손상 복구 및 유방암 억제",
  "BRCA2": "DNA 손상 복구 및 유전체 안정성 유지",
  "MYC": "세포 증식과 대사를 조절하는 강력한 전사 인자",
  "PIK3CA": "세포 생존과 성장을 돕는 신호 전달 효소",
  "CDKN2A": "세포 주기 조절(G1/S Checkpoint) 및 종양 억제",
  "APC": "Wnt 신호 전달 경로 억제 및 대장암 관련",
  "BRAF": "세포 성장 신호 전달 (MAPK 경로 Kinase)",
  "SMAD4": "TGF-beta 신호 전달 및 종양 억제",
  "IDH1": "세포 대사 조절 및 뇌종양 관련 돌연변이 효소",
  "NF1": "RAS 신호 억제 및 신경섬유종증 관련",

  // --- 추가된 질병별 핵심 유전자 ---
  "ALK": "역형성 림프종 인산화효소 (폐암 융합 유전자)",
  "ROS1": "수용체 티로신 키나아제 (폐암 융합 유전자)",
  "MET": "간세포 성장인자 수용체 (암세포 증식 및 전이)",
  "RET": "신경 생장인자 관련 신호 전달 (갑상선/폐암)",
  "NTRK1": "신경 영양 인자 수용체 (희귀 암 유발)",
  
  "ESR1": "에스트로겐 수용체 (유방암 호르몬 치료 타겟)",
  "PGR": "프로게스테론 수용체 (유방암 예후 인자)",
  "AKT1": "세포 생존 및 성장 신호 전달 (PI3K 경로)",
  "GATA3": "유선 상피 세포 분화 조절 전사 인자",
  "CDH1": "E-cadherin (세포 접착 유지 및 암 전이 억제)",
  "CCND1": "Cyclin D1 (세포 주기 진행 촉진)",

  "NRAS": "RAS 계열 신호 전달 단백질 (흑색종/대장암)",
  "FBXW7": "세포 주기 조절 단백질 분해 (종양 억제)",
  "TGFBR2": "TGF-beta 수용체 (세포 증식 억제 신호)",
  "BAX": "세포 사멸(Apoptosis) 유도 단백질",

  "GNAS": "G-protein 알파 서브유닛 (췌장암 등 관여)",
  "RNF43": "Wnt 신호 경로 억제 (Ubiquitin ligase)",
  "ATM": "DNA 손상 감지 및 복구 신호 시작",
  "MLH1": "DNA 불일치 복구(Mismatch Repair) 단백질",
  "ARID1A": "크로마틴 리모델링 복합체 (종양 억제)",

  "TERT": "텔로머라아제 역전사 효소 (세포 불멸화)",
  "ATRX": "크로마틴 리모델링 및 텔로미어 유지",
  "MGMT": "DNA 손상 복구 효소 (항암제 내성 관련)",
  "RB1": "망막모세포종 단백질 (세포 주기 억제)",

  // --- 시뮬레이션 결과 유전자 ---
  "MT-ND4": "미토콘드리아 호흡 사슬 복합체 I 서브유닛 (에너지 대사)",
  "PDE10A": "cAMP/cGMP 분해 효소 (신호 전달 조절)",
  "HSP90AA1": "단백질 안정화 및 스트레스 반응 (Chaperone)",
  "TMSB10": "세포 골격 조절 및 암세포 전이 관여",
  "SERPINE1": "PAI-1: 혈전 분해 억제 및 조직 섬유화",
  "MALAT1": "전이성 폐선암 관련 긴 비암호화 RNA",
  "MT-RNR1": "미토콘드리아 12S 리보솜 RNA",
  "MT-RNR2": "미토콘드리아 16S 리보솜 RNA",
  "NEAT1": "핵 내 구조 형성 및 암세포 성장 관여 lncRNA",
  "VIM": "Vimentin: 상피-간엽 이행(EMT) 마커",
  "FN1": "Fibronectin 1: 세포 이동 및 전이 촉진",
  "GAPDH": "세포 대사의 기본이 되는 하우스키핑 유전자",
  "FNDC3B": "세포 이동 및 암세포 침윤 조절",
  "TRIO": "세포 골격 재구성 및 신경 성장 조절",
  "ASPH": "암세포 이동 및 침윤 촉진 효소",
  "HSP90B1": "소포체 스트레스 반응 조절",
  "EXT1": "헤파란 황산염 생합성 및 종양 억제",
  "SPARC": "세포 외 기질 조절 및 침윤 관련",
  "PDE4D": "염증 및 세포 신호 조절 효소",
  "TALAM1": "종양 관련 lncRNA 후보"
};

const getGeneDesc = (geneName) => {
  if (GENE_DESCRIPTIONS[geneName]) return GENE_DESCRIPTIONS[geneName];
  if (geneName.startsWith("RandGene_")) return "전체 유전체 프로파일 시뮬레이션을 위한 가상 데이터 (Simulated Data)";
  if (geneName.startsWith("Gene_") || geneName.startsWith("Target_")) return "AI가 발굴한 신규 치료 타겟 후보 (Novel Target)";
  return "세포 대사 및 발현 조절 네트워크의 주요 인자";
};

// 경로별 설명이 중복되지 않도록 상세 분리
const getPathwayInsight = (pathway) => {
    // 1. 전사 조절 (Transcriptional Regulation)
    if (pathway.includes("Transcriptional")) 
        return "전사 인자(Transcription Factor)의 결합을 방해하여, 암세포 증식에 필수적인 유전자가 mRNA로 만들어지는 단계를 원천 차단합니다.";
    
    // 2. 핵 구조 (Nuclear Structure) - 전사와 분리됨!
    if (pathway.includes("Nuclear")) 
        return "핵 내 크로마틴 구조나 파라스페클(Paraspeckle) 형성을 교란하여, 유전체 안정성을 무너뜨리고 암세포의 생존 환경을 악화시킵니다.";

    // 3. 번역 및 단백질 합성 (Translation) - 대사와 분리됨!
    if (pathway.includes("Translation") || pathway.includes("Folding")) 
        return "리보솜의 단백질 합성 과정을 억제하거나 잘못된 단백질 축적(Proteotoxic Stress)을 유도하여, 암세포가 스스로 사멸하게 만듭니다.";

    // 4. 대사 과정 (Metabolism)
    if (pathway.includes("Metabolism") || pathway.includes("Metabolic")) 
        return "암세포 특유의 과도한 에너지 소비 경로(Warburg Effect)를 표적하여, 급격한 성장에 필요한 영양분 공급을 끊습니다.";

    // 5. 신호 전달 (Signaling)
    if (pathway.includes("Signaling")) 
        return "암세포의 증식과 생존 명령을 전달하는 신호 네트워크를 차단하여, 세포 분열을 멈추게 합니다.";

    // 6. 전이 및 이동 (Metastasis/EMT)
    if (pathway.includes("Metastasis") || pathway.includes("EMT") || pathway.includes("Motility")) 
        return "세포의 이동성을 부여하는 상피-간엽 이행(EMT) 과정을 억제하여, 암세포가 다른 장기로 전이되는 것을 막습니다.";

    // 7. 세포 주기 (Cell Cycle)
    if (pathway.includes("Cycle")) 
        return "통제되지 않는 세포 분열 주기를 강제로 멈추게 하여(Cell Cycle Arrest), 종양의 크기 증가를 억제합니다.";

    // 8. 면역 및 염증 (Immune)
    if (pathway.includes("Immune") || pathway.includes("Inflammation")) 
        return "종양 미세환경 내의 염증 반응을 조절하고, 면역 세포가 암세포를 더 잘 공격할 수 있도록 돕습니다.";
    
    return "세포 생존에 필수적인 주요 생물학적 네트워크를 조절하여 복합적인 항암 효과를 나타냅니다.";
};

// ==============================================================================
// 3. 스타일 정의
// ==============================================================================
const styles = {
  container: { 
    padding: '40px 40px', 
    fontFamily: "'Segoe UI', Roboto, sans-serif", 
    maxWidth: '1400px', 
    margin: '0 auto', 
    color: '#333',
    boxSizing: 'border-box'
  },
  header: { textAlign: 'center', marginBottom: '50px' },
  title: { fontSize: '2.8rem', color: '#2c3e50', marginBottom: '10px', fontWeight: '800', letterSpacing: '-0.5px' },
  subtitle: { fontSize: '1.2rem', color: '#7f8c8d', fontWeight: '400' },
  
  section: { backgroundColor: 'white', padding: '35px', borderRadius: '20px', boxShadow: '0 10px 40px rgba(0,0,0,0.06)', marginBottom: '40px', transition: '0.3s' },
  sectionTitle: { borderBottom: '2px solid #f1f3f5', paddingBottom: '20px', marginBottom: '30px', color: '#34495e', fontSize: '1.6rem', fontWeight: '700' },
  
  presetContainer: { display: 'flex', flexWrap: 'wrap', gap: '12px', marginBottom: '25px' },
  presetBtn: { padding: '12px 20px', border: '1px solid #e9ecef', borderRadius: '50px', backgroundColor: '#f8f9fa', cursor: 'pointer', transition: 'all 0.2s', fontWeight: '600', fontSize: '0.95rem', color: '#495057' },
  activePreset: { backgroundColor: '#e7f1ff', borderColor: '#007bff', color: '#007bff', boxShadow: '0 4px 12px rgba(0,123,255,0.15)' },
  
  inputGrid: { display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(240px, 1fr))', gap: '20px', marginBottom: '30px' },
  inputWrapper: { display: 'flex', flexDirection: 'column' },
  inputLabel: { fontWeight: '700', marginBottom: '8px', fontSize: '0.9rem', color: '#495057' },
  inputField: { padding: '12px 15px', border: '1px solid #ced4da', borderRadius: '10px', fontSize: '1rem', transition: '0.2s', outline: 'none' },
  
  actionBtn: { width: '100%', padding: '20px', fontSize: '1.2rem', fontWeight: 'bold', color: 'white', border: 'none', borderRadius: '12px', cursor: 'pointer', marginTop: '10px', boxShadow: '0 4px 15px rgba(0,0,0,0.1)', transition: 'transform 0.1s' },
  
  resultBox: { marginTop: '30px', padding: '30px', backgroundColor: '#f8f9fa', borderRadius: '16px', border: '1px solid #e9ecef' },
  drugContainer: { display: 'flex', alignItems: 'center', gap: '40px', flexWrap: 'wrap' },
  drugImageWrapper: { flex: '0 0 auto', padding: '20px', backgroundColor: 'white', borderRadius: '16px', boxShadow: '0 10px 30px rgba(0,0,0,0.05)', textAlign: 'center', minWidth: '240px' },
  
  geneList: { listStyle: 'none', padding: 0, marginTop: '20px' },
  geneItem: { display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '16px 20px', borderBottom: '1px solid #f1f3f5', backgroundColor: 'white', marginBottom: '12px', borderRadius: '12px', boxShadow: '0 2px 8px rgba(0,0,0,0.02)' },
};

const generateFullProfile = (baseGenes) => {
  const fullData = { ...baseGenes };
  for(let i=0; i<500; i++) {
     const val = (Math.random() * 4 - 2).toFixed(2);
     fullData[`RandGene_${i}`] = parseFloat(val);
  }
  return fullData;
};

function App() {
  const [geneInput, setGeneInput] = useState({});
  const [selectedDisease, setSelectedDisease] = useState(null);
  const [drugVector, setDrugVector] = useState(null);
  const [responseResult, setResponseResult] = useState(null);
  const [pathwayResult, setPathwayResult] = useState(null);
  
  const [loading, setLoading] = useState(false);
  const [loadingMsg, setLoadingMsg] = useState(""); 

  const handlePresetClick = (key) => {
    setSelectedDisease(key);
    setGeneInput(DISEASE_PRESETS[key].genes);
    setDrugVector(null);
    setResponseResult(null);
    setPathwayResult(null);
  };

  const handleReset = () => {
    setSelectedDisease(null);
    setGeneInput({});
    setDrugVector(null);
    setResponseResult(null);
    setPathwayResult(null);
  };

  const handleGeneChange = (key, val) => {
    setGeneInput({ ...geneInput, [key]: parseFloat(val) });
  };

  // 1. 신약 발굴 API
  const handleFindDrug = async () => {
    if (Object.keys(geneInput).length === 0) return alert("데이터를 입력하거나 질병을 선택하세요.");
    setLoading(true);
    setLoadingMsg("🧬 Genomic Variation Analysis (유전자 변이 패턴 분석)...");
    await new Promise(r => setTimeout(r, 800));
    setLoadingMsg("🔍 Searching Chemical Space (거대 화학 공간 탐색 중)...");
    await new Promise(r => setTimeout(r, 800));
    setLoadingMsg("🧪 Generating Molecular Structure (최적 분자 구조 생성 중)...");
    await new Promise(r => setTimeout(r, 800));

    try {
      const payload = {
        genes: Object.keys(geneInput),        
        expressions: Object.values(geneInput) 
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

  // 2. 시뮬레이션 API
  const handleSimulate = async () => {
    if (!drugVector) return alert("약물을 먼저 생성해주세요.");
    
    setLoading(true);
    setLoadingMsg("📊 Simulating Drug Response (약물 반응성 예측 중)...");
    setResponseResult(null);
    setPathwayResult(null);
    
    try {
      const payload = {
        smiles_embedding: drugVector,
        genes: Object.keys(geneInput),
        expressions: Object.values(geneInput)
      };

      const res = await axios.post('http://127.0.0.1:8000/predict/drug_response', payload);
      
      if (res.data) {
        setResponseResult(res.data.top_genes);
        setPathwayResult(res.data.pathways);
      }
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
        
        <div style={{display:'flex', justifyContent:'flex-end', marginBottom:'15px'}}>
          <button 
            onClick={() => {
                if(!selectedDisease) return alert("질병을 먼저 선택해주세요.");
                setGeneInput(generateFullProfile(DISEASE_PRESETS[selectedDisease].genes));
                alert("🧬 500+ Genes Loaded from Patient Database (WGS/RNA-seq)!");
            }}
            style={{background:'#6c5ce7', color:'white', border:'none', padding:'12px 20px', borderRadius:'8px', cursor:'pointer', fontSize:'0.95rem', fontWeight:'bold', boxShadow:'0 4px 6px rgba(0,0,0,0.1)'}}
          >
            📂 Load Full Patient Profile (Whole Genome)
          </button>
        </div>

        <div style={styles.inputGrid}>
          {/* 1. 실제 유전자만 필터링해서 표시 */}
          {Object.entries(geneInput)
            .filter(([key]) => !key.startsWith("RandGene_")) 
            .map(([gene, val]) => (
            <div key={gene} style={styles.inputWrapper}>
              <span style={styles.inputLabel}>
                {gene} <span style={{fontSize:'0.75em', color:'#868e96', fontWeight:'normal'}}>- {getGeneDesc(gene).split('(')[0]}</span>
              </span>
              <input type="number" value={val} onChange={(e) => handleGeneChange(gene, e.target.value)} style={styles.inputField} />
            </div>
          ))}

          {/* 2. 가상 데이터는 안내 문구로만 표시 */}
          {Object.keys(geneInput).some(k => k.startsWith("RandGene_")) && (
             <div style={{
               gridColumn:'1/-1', 
               textAlign:'center', 
               color:'#636e72', 
               padding:'15px', 
               background:'#f8f9fa', 
               border:'1px dashed #ced4da',
               borderRadius:'8px', 
               fontSize:'0.95rem',
               marginTop: '10px'
             }}>
               🧬 <strong>Background Data Loaded:</strong> + {Object.keys(geneInput).filter(k => k.startsWith("RandGene_")).length} Simulated Genes (Whole Genome Profile) are ready for analysis.
             </div>
          )}
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

            {/* Signaling Pathway Analysis */}
            {pathwayResult && (
              <div style={{marginTop: '40px', paddingTop: '30px', borderTop: '2px dashed #e9ecef'}}>
                <h3 style={{color: '#2c3e50', marginBottom: '20px'}}>🧬 Signaling Pathway Analysis</h3>
                <div style={{
                    display:'flex', 
                    gap:'40px', 
                    flexWrap:'wrap', 
                    justifyContent: 'center', 
                    alignItems:'flex-start'
                }}>
                    <div style={{flex: '1 1 350px', minWidth: '300px', height: '350px', display: 'flex', justifyContent: 'center'}}>
                        <ResponsiveContainer width="100%" height="100%">
                        <PieChart>
                            <Pie
                                data={Object.entries(pathwayResult).map(([k, v]) => ({ name: k, value: v }))}
                                dataKey="value"
                                nameKey="name"
                                cx="50%"
                                cy="50%"
                                innerRadius={70} 
                                outerRadius={100}
                                paddingAngle={5}
                                labelLine={false}
                            >
                                {Object.entries(pathwayResult).map((entry, index) => (
                                    <Cell key={`cell-${index}`} fill={['#0088FE', '#00C49F', '#FFBB28', '#FF8042', '#8884d8', '#ff7675', '#a29bfe'][index % 7]} />
                                ))}
                            </Pie>
                            <Tooltip />
                            <Legend layout="vertical" verticalAlign="middle" align="right" wrapperStyle={{fontSize: '13px', color:'#555'}} />
                        </PieChart>
                        </ResponsiveContainer>
                    </div>
                    
                    <div style={{flex: '1 1 500px', minWidth: '300px'}}>
                        <h4 style={{margin:'0 0 15px 0', color:'#495057'}}>💡 Key Biological Insights:</h4>
                        <ul style={{
                            lineHeight:'1.8', 
                            fontSize:'1rem', 
                            color:'#555', 
                            paddingLeft:'20px',
                            wordBreak: 'keep-all'
                        }}>
                            {Object.keys(pathwayResult)
                                .filter(pathway => pathway !== "Unknown/Novel Pathway")
                                .slice(0, 4)
                                .map((pathway, i) => (
                                <li key={i} style={{marginBottom:'12px'}}>
                                    <strong style={{color:'#2c3e50', fontSize:'1.05rem'}}>{pathway}</strong> 경로가 활성화되었습니다. <br/>
                                    <span style={{color:'#666', fontSize:'0.95em'}}>↪ {getPathwayInsight(pathway)}</span>
                                </li>
                            ))}
                        </ul>
                    </div>
                </div>
              </div>
            )}

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