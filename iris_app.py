"""
Iris Classifier – Full-Stack Flask App
Run: pip install flask scikit-learn numpy pandas joblib
     python iris_app.py
"""

import warnings, json, os
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib

from flask import Flask, request, jsonify, render_template_string

warnings.filterwarnings("ignore")
np.random.seed(42)

# ── Train / Load model ─────────────────────────────────────────────────────
MODEL_PATH = "iris_classifier_nb.joblib"
iris_data   = load_iris()
df = pd.DataFrame(iris_data.data, columns=iris_data.feature_names)
df["species"] = iris_data.target
df["species"] = df["species"].map({0:"setosa",1:"versicolor",2:"virginica"})

X, y = df.drop("species", axis=1), df["species"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

model = GaussianNB()
model.fit(X_train, y_train)
joblib.dump(model, MODEL_PATH)

y_pred  = model.predict(X_test)
acc     = accuracy_score(y_test, y_pred)
cv      = cross_val_score(model, X, y, cv=5)
cm      = confusion_matrix(y_test, y_pred).tolist()
report  = classification_report(y_test, y_pred, output_dict=True)

STATS = {
    "accuracy":   f"{acc:.2%}",
    "cv_mean":    f"{cv.mean():.2%}",
    "cv_std":     f"{cv.std():.2%}",
    "cm":         cm,
    "report":     report,
    "classes":    list(iris_data.target_names),
    "train_size": len(X_train),
    "test_size":  len(X_test),
}

# ── HTML Template ──────────────────────────────────────────────────────────
HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>Iris · Neural Classifier</title>
<link rel="preconnect" href="https://fonts.googleapis.com"/>
<link href="https://fonts.googleapis.com/css2?family=Space+Mono:ital,wght@0,400;0,700;1,400&family=Syne:wght@400;700;800&display=swap" rel="stylesheet"/>
<style>
/* ─── TOKENS ─────────────────────────────────────── */
:root{
  --bg:#05070f;
  --surface:#0c1120;
  --border:#1a2540;
  --accent:#4ff0c0;
  --accent2:#ff6b6b;
  --accent3:#ffd166;
  --text:#e8edf5;
  --muted:#5a6a8a;
  --setosa:#4ff0c0;
  --versicolor:#ffd166;
  --virginica:#ff6b6b;
  --radius:12px;
  --mono:'Space Mono',monospace;
  --sans:'Syne',sans-serif;
}
*{box-sizing:border-box;margin:0;padding:0}
html{scroll-behavior:smooth}
body{
  background:var(--bg);
  color:var(--text);
  font-family:var(--mono);
  min-height:100vh;
  overflow-x:hidden;
}

/* ─── BACKGROUND GRID ────────────────────────────── */
body::before{
  content:'';
  position:fixed;inset:0;
  background-image:
    linear-gradient(var(--border) 1px, transparent 1px),
    linear-gradient(90deg, var(--border) 1px, transparent 1px);
  background-size:40px 40px;
  opacity:.35;
  pointer-events:none;
  z-index:0;
}

/* ─── HEADER ─────────────────────────────────────── */
header{
  position:relative;z-index:10;
  padding:3rem 2rem 2rem;
  text-align:center;
  border-bottom:1px solid var(--border);
  background:linear-gradient(180deg,rgba(79,240,192,.06) 0%,transparent 100%);
}
.logo-badge{
  display:inline-block;
  background:var(--accent);
  color:var(--bg);
  font-family:var(--mono);
  font-size:.65rem;
  font-weight:700;
  letter-spacing:.15em;
  padding:.35rem .8rem;
  border-radius:4px;
  margin-bottom:1.2rem;
  text-transform:uppercase;
}
h1{
  font-family:var(--sans);
  font-size:clamp(2.2rem,6vw,4.5rem);
  font-weight:800;
  letter-spacing:-.03em;
  line-height:1;
  background:linear-gradient(135deg,var(--accent),#a8fce0,var(--accent));
  background-size:200%;
  -webkit-background-clip:text;
  -webkit-text-fill-color:transparent;
  animation:shimmer 4s linear infinite;
}
@keyframes shimmer{to{background-position:200% center}}
.tagline{
  margin-top:.8rem;
  color:var(--muted);
  font-size:.8rem;
  letter-spacing:.1em;
  text-transform:uppercase;
}

/* ─── STAT STRIP ─────────────────────────────────── */
.stat-strip{
  position:relative;z-index:10;
  display:flex;flex-wrap:wrap;
  gap:1px;
  background:var(--border);
  border-bottom:1px solid var(--border);
}
.stat-block{
  flex:1;min-width:140px;
  background:var(--surface);
  padding:1.2rem 1.5rem;
  text-align:center;
}
.stat-label{
  font-size:.65rem;
  letter-spacing:.12em;
  text-transform:uppercase;
  color:var(--muted);
  margin-bottom:.4rem;
}
.stat-value{
  font-family:var(--sans);
  font-size:1.8rem;
  font-weight:800;
  color:var(--accent);
}
.stat-sub{font-size:.65rem;color:var(--muted);margin-top:.2rem;}

/* ─── MAIN GRID ──────────────────────────────────── */
main{
  position:relative;z-index:10;
  display:grid;
  grid-template-columns:1fr 1fr;
  gap:1px;
  background:var(--border);
  min-height:calc(100vh - 280px);
}
@media(max-width:800px){main{grid-template-columns:1fr}}

/* ─── PANEL ──────────────────────────────────────── */
.panel{
  background:var(--surface);
  padding:2rem;
}
.panel-title{
  font-family:var(--sans);
  font-size:.7rem;
  font-weight:700;
  letter-spacing:.18em;
  text-transform:uppercase;
  color:var(--muted);
  border-bottom:1px solid var(--border);
  padding-bottom:.8rem;
  margin-bottom:1.5rem;
  display:flex;align-items:center;gap:.5rem;
}
.panel-title span{color:var(--accent)}

/* ─── SLIDERS ────────────────────────────────────── */
.feature-grid{display:flex;flex-direction:column;gap:1.4rem;}
.feature-row label{
  display:flex;justify-content:space-between;align-items:baseline;
  font-size:.72rem;color:var(--text);letter-spacing:.04em;
  margin-bottom:.5rem;
}
.feat-name{text-transform:uppercase;letter-spacing:.1em;color:var(--muted)}
.feat-val{
  font-family:var(--sans);
  font-weight:700;
  font-size:1rem;
  color:var(--accent);
  min-width:40px;
  text-align:right;
}
input[type=range]{
  -webkit-appearance:none;
  appearance:none;
  width:100%;
  height:4px;
  background:var(--border);
  border-radius:2px;
  outline:none;
  cursor:pointer;
  transition:background .2s;
}
input[type=range]::-webkit-slider-thumb{
  -webkit-appearance:none;
  width:18px;height:18px;
  border-radius:50%;
  background:var(--accent);
  box-shadow:0 0 0 4px rgba(79,240,192,.15);
  cursor:pointer;
  transition:box-shadow .2s,transform .1s;
}
input[type=range]:hover::-webkit-slider-thumb{
  box-shadow:0 0 0 8px rgba(79,240,192,.2);
  transform:scale(1.15);
}
.range-labels{
  display:flex;justify-content:space-between;
  font-size:.55rem;color:var(--muted);margin-top:.2rem;
}

/* ─── PREDICT BUTTON ─────────────────────────────── */
.btn-predict{
  width:100%;
  margin-top:1.8rem;
  padding:1rem 2rem;
  background:var(--accent);
  color:var(--bg);
  border:none;
  border-radius:var(--radius);
  font-family:var(--sans);
  font-size:1rem;
  font-weight:800;
  letter-spacing:.08em;
  cursor:pointer;
  transition:transform .15s, box-shadow .15s, background .15s;
  position:relative;overflow:hidden;
}
.btn-predict::after{
  content:'';
  position:absolute;inset:0;
  background:linear-gradient(135deg,rgba(255,255,255,.15),transparent);
  opacity:0;
  transition:opacity .2s;
}
.btn-predict:hover{
  transform:translateY(-2px);
  box-shadow:0 8px 32px rgba(79,240,192,.35);
}
.btn-predict:hover::after{opacity:1}
.btn-predict:active{transform:translateY(0);box-shadow:none}
.btn-predict.loading{
  pointer-events:none;
  opacity:.7;
}

/* ─── RESULT CARD ────────────────────────────────── */
.result-area{
  min-height:200px;
  display:flex;flex-direction:column;
  gap:1rem;
}
.result-card{
  border:1px solid var(--border);
  border-radius:var(--radius);
  padding:1.5rem;
  background:rgba(255,255,255,.02);
  opacity:0;
  transform:translateY(12px);
  transition:opacity .4s ease, transform .4s ease;
}
.result-card.show{opacity:1;transform:translateY(0);}
.species-badge{
  display:inline-block;
  font-family:var(--sans);
  font-size:2rem;
  font-weight:800;
  letter-spacing:-.02em;
  margin-bottom:.6rem;
}
.species-badge.setosa    {color:var(--setosa);}
.species-badge.versicolor{color:var(--versicolor);}
.species-badge.virginica {color:var(--virginica);}

.confidence-row{
  font-size:.7rem;color:var(--muted);
  text-transform:uppercase;letter-spacing:.1em;
  margin-bottom:.8rem;
}
.confidence-row strong{color:var(--text);font-size:1rem;font-family:var(--sans)}

/* ─── PROB BARS ──────────────────────────────────── */
.prob-bars{display:flex;flex-direction:column;gap:.6rem;margin-top:.5rem;}
.prob-row{font-size:.68rem;}
.prob-label{
  display:flex;justify-content:space-between;
  color:var(--muted);letter-spacing:.08em;text-transform:uppercase;
  margin-bottom:.25rem;
}
.prob-track{
  height:6px;
  background:var(--border);
  border-radius:3px;
  overflow:hidden;
}
.prob-fill{
  height:100%;
  border-radius:3px;
  width:0%;
  transition:width .8s cubic-bezier(.4,0,.2,1);
}
.prob-fill.setosa    {background:var(--setosa);}
.prob-fill.versicolor{background:var(--versicolor);}
.prob-fill.virginica {background:var(--virginica);}

/* ─── CONFUSION MATRIX ───────────────────────────── */
.cm-grid{
  display:grid;
  grid-template-columns:auto repeat(3,1fr);
  gap:3px;
  margin-top:.5rem;
}
.cm-cell{
  padding:.55rem .4rem;
  text-align:center;
  font-size:.75rem;
  border-radius:4px;
  background:var(--bg);
}
.cm-cell.header{
  background:transparent;
  color:var(--muted);
  font-size:.6rem;
  letter-spacing:.08em;
  text-transform:uppercase;
  display:flex;align-items:center;justify-content:center;
}
.cm-cell.diagonal{
  font-family:var(--sans);
  font-weight:700;
  font-size:1rem;
}
.axis-label{
  writing-mode:vertical-rl;
  text-orientation:mixed;
  font-size:.55rem;
  letter-spacing:.1em;
  text-transform:uppercase;
  color:var(--muted);
  text-align:center;
}

/* ─── SAMPLE GALLERY ─────────────────────────────── */
.samples{display:flex;flex-direction:column;gap:.7rem;}
.sample-btn{
  display:flex;align-items:center;gap:1rem;
  padding:.8rem 1rem;
  background:var(--bg);
  border:1px solid var(--border);
  border-radius:var(--radius);
  cursor:pointer;
  transition:border-color .2s,background .2s;
  text-align:left;
  width:100%;
}
.sample-btn:hover{border-color:var(--accent);background:rgba(79,240,192,.04);}
.sample-dot{
  width:10px;height:10px;
  border-radius:50%;
  flex-shrink:0;
}
.sample-dot.setosa    {background:var(--setosa);}
.sample-dot.versicolor{background:var(--versicolor);}
.sample-dot.virginica {background:var(--virginica);}
.sample-name{
  font-family:var(--sans);
  font-weight:700;
  font-size:.85rem;
  color:var(--text);
  flex:1;
}
.sample-vals{
  font-size:.62rem;
  color:var(--muted);
  letter-spacing:.04em;
}

/* ─── CLASS REPORT TABLE ─────────────────────────── */
.report-table{
  width:100%;
  border-collapse:collapse;
  font-size:.7rem;
}
.report-table th{
  font-size:.6rem;
  letter-spacing:.12em;
  text-transform:uppercase;
  color:var(--muted);
  padding:.5rem .8rem;
  border-bottom:1px solid var(--border);
  text-align:left;
}
.report-table td{
  padding:.55rem .8rem;
  border-bottom:1px solid rgba(255,255,255,.04);
}
.report-table tr:last-child td{border-bottom:none;}
.badge-pill{
  display:inline-block;
  padding:.2rem .55rem;
  border-radius:20px;
  font-size:.6rem;
  font-weight:700;
  letter-spacing:.06em;
}
.badge-pill.setosa    {background:rgba(79,240,192,.12);color:var(--setosa);}
.badge-pill.versicolor{background:rgba(255,209,102,.12);color:var(--versicolor);}
.badge-pill.virginica {background:rgba(255,107,107,.12);color:var(--virginica);}

/* ─── HISTORY LOG ────────────────────────────────── */
.history-log{
  max-height:260px;
  overflow-y:auto;
  display:flex;flex-direction:column;gap:.5rem;
  padding-right:.3rem;
}
.history-log::-webkit-scrollbar{width:3px;}
.history-log::-webkit-scrollbar-track{background:transparent;}
.history-log::-webkit-scrollbar-thumb{background:var(--border);border-radius:2px;}
.hist-entry{
  display:flex;align-items:center;gap:.8rem;
  padding:.6rem .8rem;
  background:var(--bg);
  border-radius:8px;
  font-size:.65rem;
  color:var(--muted);
  animation:slideIn .3s ease;
}
@keyframes slideIn{from{opacity:0;transform:translateX(-8px)}to{opacity:1;transform:none}}
.hist-dot{width:8px;height:8px;border-radius:50%;flex-shrink:0;}
.hist-dot.setosa    {background:var(--setosa);}
.hist-dot.versicolor{background:var(--versicolor);}
.hist-dot.virginica {background:var(--virginica);}
.hist-species{
  font-family:var(--sans);
  font-weight:700;
  font-size:.8rem;
  color:var(--text);
  min-width:90px;
}
.hist-conf{margin-left:auto;font-size:.65rem;}
.empty-state{
  color:var(--muted);
  font-size:.72rem;
  text-align:center;
  padding:2rem 0;
  letter-spacing:.06em;
}

/* ─── TOAST ──────────────────────────────────────── */
.toast{
  position:fixed;bottom:2rem;right:2rem;
  background:var(--accent);
  color:var(--bg);
  font-family:var(--sans);
  font-weight:700;
  font-size:.8rem;
  padding:.7rem 1.2rem;
  border-radius:8px;
  z-index:999;
  transform:translateY(20px);
  opacity:0;
  transition:all .3s ease;
  pointer-events:none;
}
.toast.show{transform:translateY(0);opacity:1;}

footer{
  position:relative;z-index:10;
  text-align:center;
  padding:1.2rem;
  font-size:.6rem;
  letter-spacing:.1em;
  text-transform:uppercase;
  color:var(--muted);
  border-top:1px solid var(--border);
}
</style>
</head>
<body>

<!-- ─── HEADER ────────────────────────────────────── -->
<header>
  <div class="logo-badge">Gaussian Naive Bayes · Iris Dataset</div>
  <h1>Iris Classifier</h1>
  <p class="tagline">Real-time species prediction · scikit-learn model</p>
</header>

<!-- ─── STAT STRIP ─────────────────────────────────── -->
<div class="stat-strip">
  <div class="stat-block">
    <div class="stat-label">Test Accuracy</div>
    <div class="stat-value" id="acc">–</div>
    <div class="stat-sub">held-out 20%</div>
  </div>
  <div class="stat-block">
    <div class="stat-label">CV Accuracy (5-fold)</div>
    <div class="stat-value" id="cv">–</div>
    <div class="stat-sub" id="cv-std">±–</div>
  </div>
  <div class="stat-block">
    <div class="stat-label">Train / Test Split</div>
    <div class="stat-value" id="split">–</div>
    <div class="stat-sub">samples</div>
  </div>
  <div class="stat-block">
    <div class="stat-label">Classes</div>
    <div class="stat-value">3</div>
    <div class="stat-sub">setosa · versicolor · virginica</div>
  </div>
</div>

<!-- ─── MAIN ───────────────────────────────────────── -->
<main>

  <!-- LEFT COL ─── Panel 1: Predictor -->
  <div class="panel">
    <div class="panel-title"><span>⬡</span> Feature Input</div>
    <div class="feature-grid" id="sliders"></div>
    <button class="btn-predict" id="predictBtn" onclick="predict()">
      Classify Iris →
    </button>
  </div>

  <!-- RIGHT COL ─── Panel 2: Result -->
  <div class="panel">
    <div class="panel-title"><span>◈</span> Prediction Result</div>
    <div class="result-area">
      <div class="result-card" id="resultCard">
        <div class="species-badge" id="speciesBadge">–</div>
        <div class="confidence-row">
          Confidence: <strong id="confVal">–</strong>
        </div>
        <div class="prob-bars" id="probBars"></div>
      </div>
      <div class="empty-state" id="emptyState">
        Adjust sliders and hit Classify →
      </div>
    </div>
  </div>

  <!-- LEFT COL ─── Panel 3: Quick Samples -->
  <div class="panel">
    <div class="panel-title"><span>◉</span> Quick Samples</div>
    <div class="samples" id="sampleBtns"></div>
  </div>

  <!-- RIGHT COL ─── Panel 4: Confusion Matrix -->
  <div class="panel">
    <div class="panel-title"><span>⊞</span> Confusion Matrix</div>
    <div class="cm-grid" id="cmGrid"></div>
  </div>

  <!-- LEFT COL ─── Panel 5: Classification Report -->
  <div class="panel">
    <div class="panel-title"><span>≡</span> Classification Report</div>
    <table class="report-table" id="reportTable"></table>
  </div>

  <!-- RIGHT COL ─── Panel 6: Prediction History -->
  <div class="panel">
    <div class="panel-title"><span>↺</span> Prediction History</div>
    <div class="history-log" id="historyLog">
      <div class="empty-state">No predictions yet.</div>
    </div>
  </div>

</main>

<footer>Iris Neural Classifier · Gaussian Naive Bayes · scikit-learn</footer>
<div class="toast" id="toast"></div>

<script>
/* ── Config ───────────────────────────────────────── */
const FEATURES = [
  {name:"Sepal Length",key:"sepal length (cm)",min:4.0,max:8.0,def:5.8,step:.1},
  {name:"Sepal Width", key:"sepal width (cm)", min:2.0,max:4.5,def:3.1,step:.1},
  {name:"Petal Length",key:"petal length (cm)",min:1.0,max:7.0,def:3.8,step:.1},
  {name:"Petal Width", key:"petal width (cm)", min:0.1,max:2.5,def:1.2,step:.1},
];
const SAMPLES = [
  {label:"Typical Setosa",    cls:"setosa",    vals:[5.1,3.5,1.4,0.2]},
  {label:"Typical Versicolor",cls:"versicolor",vals:[6.0,2.7,5.1,1.6]},
  {label:"Typical Virginica", cls:"virginica", vals:[6.5,3.0,5.5,1.8]},
  {label:"Borderline Case",   cls:"versicolor",vals:[5.7,2.8,4.5,1.3]},
  {label:"Large Setosa",      cls:"setosa",    vals:[5.4,3.9,1.7,0.4]},
];
const COLOR = {setosa:"#4ff0c0",versicolor:"#ffd166",virginica:"#ff6b6b"};

/* ── Build sliders ────────────────────────────────── */
const sliderDiv = document.getElementById("sliders");
FEATURES.forEach(f=>{
  sliderDiv.innerHTML += `
  <div class="feature-row">
    <label>
      <span class="feat-name">${f.name}</span>
      <span class="feat-val" id="val_${f.key}">${f.def.toFixed(1)} cm</span>
    </label>
    <input type="range" id="sl_${f.key}"
      min="${f.min}" max="${f.max}" step="${f.step}" value="${f.def}"
      oninput="document.getElementById('val_${f.key}').textContent=parseFloat(this.value).toFixed(1)+' cm'"/>
    <div class="range-labels"><span>${f.min}</span><span>${f.max}</span></div>
  </div>`;
});

/* ── Build quick samples ──────────────────────────── */
const sampleDiv = document.getElementById("sampleBtns");
SAMPLES.forEach(s=>{
  const btn = document.createElement("button");
  btn.className = "sample-btn";
  btn.innerHTML = `
    <span class="sample-dot ${s.cls}"></span>
    <span class="sample-name">${s.label}</span>
    <span class="sample-vals">${s.vals.join(' · ')}</span>`;
  btn.onclick = ()=>loadSample(s.vals);
  sampleDiv.appendChild(btn);
});

function loadSample(vals){
  FEATURES.forEach((f,i)=>{
    const sl = document.getElementById("sl_"+f.key);
    sl.value = vals[i];
    document.getElementById("val_"+f.key).textContent = vals[i].toFixed(1)+" cm";
  });
  predict();
}

/* ── Predict ──────────────────────────────────────── */
async function predict(){
  const btn = document.getElementById("predictBtn");
  btn.classList.add("loading");
  btn.textContent = "Classifying…";

  const features = {};
  FEATURES.forEach(f=>{
    features[f.key] = parseFloat(document.getElementById("sl_"+f.key).value);
  });

  try{
    const res = await fetch("/predict",{
      method:"POST",
      headers:{"Content-Type":"application/json"},
      body:JSON.stringify({features})
    });
    const d = await res.json();
    showResult(d);
    addHistory(d, features);
  }catch(e){
    showToast("⚠ Server error");
  }
  btn.classList.remove("loading");
  btn.textContent = "Classify Iris →";
}

function showResult(d){
  const card = document.getElementById("resultCard");
  const empty = document.getElementById("emptyState");
  const badge = document.getElementById("speciesBadge");
  const confVal= document.getElementById("confVal");
  const bars   = document.getElementById("probBars");

  card.classList.remove("show");
  empty.style.display = "none";

  setTimeout(()=>{
    badge.textContent = d.species;
    badge.className = `species-badge ${d.species}`;
    confVal.textContent = `${(d.confidence*100).toFixed(1)}%`;

    bars.innerHTML = d.probabilities.map(p=>`
      <div class="prob-row">
        <div class="prob-label">
          <span>${p.class}</span><span>${(p.probability*100).toFixed(1)}%</span>
        </div>
        <div class="prob-track">
          <div class="prob-fill ${p.class}" style="width:0%" data-w="${p.probability*100}"></div>
        </div>
      </div>`).join("");

    card.classList.add("show");
    setTimeout(()=>{
      document.querySelectorAll(".prob-fill").forEach(el=>{
        el.style.width = el.dataset.w + "%";
      });
    }, 100);
  }, 200);
}

/* ── History ──────────────────────────────────────── */
const history = [];
function addHistory(d, features){
  history.unshift({species:d.species, conf:d.confidence, features});
  const log = document.getElementById("historyLog");
  log.innerHTML = history.slice(0,20).map(h=>`
    <div class="hist-entry">
      <span class="hist-dot ${h.species}"></span>
      <span class="hist-species">${h.species}</span>
      <span>${Object.values(h.features).join(' · ')}</span>
      <span class="hist-conf">${(h.conf*100).toFixed(1)}%</span>
    </div>`).join("");
}

/* ── Toast ────────────────────────────────────────── */
function showToast(msg){
  const t = document.getElementById("toast");
  t.textContent = msg;
  t.classList.add("show");
  setTimeout(()=>t.classList.remove("show"), 2500);
}

/* ── Load stats from server ───────────────────────── */
fetch("/stats").then(r=>r.json()).then(s=>{
  document.getElementById("acc").textContent   = s.accuracy;
  document.getElementById("cv").textContent    = s.cv_mean;
  document.getElementById("cv-std").textContent= "±"+s.cv_std;
  document.getElementById("split").textContent = `${s.train_size}/${s.test_size}`;
  buildCM(s.cm, s.classes);
  buildReport(s.report, s.classes);
});

/* ── Confusion Matrix ─────────────────────────────── */
function buildCM(cm, cls){
  const el = document.getElementById("cmGrid");
  const clsShort = cls.map(c=>c.substring(0,3).toUpperCase());
  let html = `<div class="axis-label">TRUE</div>`;
  clsShort.forEach(c=>html+=`<div class="cm-cell header">${c}</div>`);
  const maxVal = Math.max(...cm.flat());
  cm.forEach((row,i)=>{
    html += `<div class="cm-cell header">${clsShort[i]}</div>`;
    row.forEach((v,j)=>{
      const alpha = 0.12 + (v/maxVal)*0.7;
      const isD   = i===j;
      const c     = cls[j];
      const bg    = isD ? `rgba(${hexToRgb(COLOR[c])},${alpha})` : `rgba(255,255,255,.03)`;
      html += `<div class="cm-cell ${isD?"diagonal":""}" style="background:${bg};color:${isD?COLOR[c]:"#8899aa"}">${v}</div>`;
    });
  });
  el.innerHTML = html;
  el.style.gridTemplateColumns=`auto repeat(${cls.length},1fr)`;
}
function hexToRgb(h){
  const r=parseInt(h.slice(1,3),16),g=parseInt(h.slice(3,5),16),b=parseInt(h.slice(5,7),16);
  return `${r},${g},${b}`;
}

/* ── Classification Report ────────────────────────── */
function buildReport(report, cls){
  const el = document.getElementById("reportTable");
  el.innerHTML = `<thead><tr>
    <th>Class</th><th>Precision</th><th>Recall</th><th>F1</th><th>Support</th>
  </tr></thead>`;
  const tbody = document.createElement("tbody");
  cls.forEach(c=>{
    const r = report[c];
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td><span class="badge-pill ${c}">${c}</span></td>
      <td>${r.precision.toFixed(3)}</td>
      <td>${r.recall.toFixed(3)}</td>
      <td>${r["f1-score"].toFixed(3)}</td>
      <td>${r.support}</td>`;
    tbody.appendChild(tr);
  });
  el.appendChild(tbody);
}
</script>
</body>
</html>
"""

# ── Flask App ──────────────────────────────────────────────────────────────
app = Flask(__name__)

@app.route("/")
def index():
    return render_template_string(HTML)

@app.route("/predict", methods=["POST"])
def predict():
    data     = request.get_json()
    features = data.get("features", {})
    df_in    = pd.DataFrame([features])
    pred     = model.predict(df_in)[0]
    probs    = model.predict_proba(df_in)[0]
    conf     = float(probs.max())
    prob_list = [{"class": cls, "probability": float(p)}
                 for cls, p in zip(model.classes_, probs)]
    return jsonify({"species": pred, "confidence": conf, "probabilities": prob_list})

@app.route("/stats")
def stats():
    return jsonify(STATS)

# ── Entry point ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "="*55)
    print("   Iris Classifier App")
    print("="*55)
    print(f"  Model accuracy : {acc:.2%}")
    print(f"   CV accuracy    : {cv.mean():.2%} ± {cv.std():.2%}")
    print(f"   Open browser   : http://127.0.0.1:5000")
    print("="*55 + "\n")
    app.run(debug=True, port=5000)
