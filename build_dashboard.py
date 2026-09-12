"""Build a self-contained, historical-data MVP dashboard for the casting line."""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score


BASE = Path(__file__).parent
CSV = BASE / "Input.csv"
OUT = BASE / "manufacturing_control_dashboard.html"
MODEL_OUT = BASE / "artifacts" / "defect_model.pkl"
THRESHOLD = 0.68
MAX_SCATTER_POINTS = 8_000

LABELS = {
    "injection_pressure": "주입 압력",
    "bottom_temp2": "하부 온도 2",
    "bottom_temp1": "하부 온도 1",
    "bottom_temp3": "하부 온도 3",
    "top_temp1": "상부 온도 1",
    "top_temp2": "상부 온도 2",
    "top_temp3": "상부 온도 3",
    "top_temp4": "상부 온도 4",
    "bottom_temp4": "하부 온도 4",
    "cooling_water_temp": "냉각수 온도",
    "sleeve_temperature": "슬리브 온도",
    "mold_temperature": "금형 온도",
    "facility_CycleTime": "설비 사이클타임",
    "production_CycleTime": "생산 사이클타임",
    "production_count": "생산 수량",
    "mechanical_strength": "기계적 강도",
    "biscuit_thickness": "비스킷 두께",
}


def load_events() -> tuple[pd.DataFrame, list[str]]:
    raw = pd.read_csv(CSV)
    raw["event_at"] = pd.to_datetime(raw["timestamp"] + " " + raw["date"], errors="coerce")
    events = raw.dropna(subset=["event_at", "PassOrFail"]).copy()
    events["PassOrFail"] = events["PassOrFail"].astype(int)
    events = events.sort_values("event_at").reset_index(drop=True)
    features = [
        col
        for col in events.select_dtypes(include=np.number).columns
        if col not in {"PassOrFail", "molten_capacity"}
    ]
    return events, features


def undersample(x: pd.DataFrame, y: np.ndarray, ratio: float = 0.5) -> tuple[pd.DataFrame, np.ndarray]:
    """Keep every defect and a reproducible subset of normal events."""
    rng = np.random.default_rng(42)
    positive = np.flatnonzero(y == 1)
    negative = np.flatnonzero(y == 0)
    keep_negative = rng.choice(negative, size=min(len(negative), int(len(positive) / ratio)), replace=False)
    keep = np.sort(np.concatenate([positive, keep_negative]))
    return x.iloc[keep], y[keep]


def fit_model(events: pd.DataFrame, features: list[str]) -> tuple[np.ndarray, dict[str, float], dict[str, float], RandomForestClassifier, dict[str, float]]:
    x = events[features].copy()
    split = int(len(x) * 0.8)
    medians = x.iloc[:split].median(numeric_only=True)
    x = x.fillna(medians).fillna(0)
    y = events["PassOrFail"].to_numpy()

    x_train, y_train = undersample(x.iloc[:split], y[:split])
    model = RandomForestClassifier(
        n_estimators=150,
        max_depth=None,
        max_features="sqrt",
        min_samples_leaf=1,
        min_samples_split=10,
        criterion="entropy",
        bootstrap=True,
        max_samples=0.8,
        n_jobs=-1,
        random_state=42,
    )
    model.fit(x_train, y_train)
    test_prob = model.predict_proba(x.iloc[split:])[:, 1]
    test_pred = (test_prob >= THRESHOLD).astype(int)
    metrics = {
        "prAuc": round(float(average_precision_score(y[split:], test_prob)), 4),
        "recall": round(float(recall_score(y[split:], test_pred, zero_division=0)), 4),
        "precision": round(float(precision_score(y[split:], test_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y[split:], test_pred, zero_division=0)), 4),
    }

    # Score all historical events using a model refit on all available history.
    full_x, full_y = undersample(x, y)
    model.fit(full_x, full_y)
    probabilities = model.predict_proba(x)[:, 1]
    importance = dict(zip(features, model.feature_importances_, strict=True))
    return probabilities, importance, metrics, model, medians.to_dict()


def feature_specs(events: pd.DataFrame, probabilities: np.ndarray, importance: dict[str, float]) -> dict[str, dict[str, float | str]]:
    ranked = sorted(importance, key=importance.get, reverse=True)
    low_risk = events.loc[probabilities < 0.30]
    specs: dict[str, dict[str, float | str]] = {}
    for position, feature in enumerate(ranked):
        values = pd.to_numeric(events[feature], errors="coerce").dropna()
        candidate = pd.to_numeric(low_risk[feature], errors="coerce").dropna()
        if len(values) < 20 or len(candidate) < 20:
            continue
        grade = "A" if position < 5 else "B" if position < 10 else "C"
        specs[feature] = {
            "label": LABELS.get(feature, feature),
            "grade": grade,
            "importance": round(float(importance[feature]), 4),
            "low": round(float(candidate.quantile(0.05)), 3),
            "high": round(float(candidate.quantile(0.95)), 3),
            "mean": round(float(values.mean()), 3),
            "std": round(float(values.std(ddof=0)), 3),
        }
    return specs


def sampled_records(events: pd.DataFrame, probabilities: np.ndarray, shown_features: list[str]) -> list[dict[str, object]]:
    # ponytail: static MVP ships the full time range but caps plot-heavy client work.
    if len(events) > MAX_SCATTER_POINTS:
        sampled = pd.concat([events.iloc[:: max(1, len(events) // MAX_SCATTER_POINTS)], events.tail(500)]).drop_duplicates()
        probabilities = probabilities[sampled.index.to_numpy()]
        events = sampled
    records: list[dict[str, object]] = []
    for row, probability in zip(events.itertuples(index=False), probabilities, strict=True):
        row_dict = row._asdict()
        values = {key: round(float(row_dict[key]), 3) if pd.notna(row_dict[key]) else None for key in shown_features}
        records.append(
            {
                "t": row_dict["event_at"].strftime("%Y-%m-%d %H:%M:%S"),
                "d": row_dict["event_at"].strftime("%Y-%m-%d"),
                "y": int(row_dict["PassOrFail"]),
                "p": round(float(probability), 4),
                "v": values,
            }
        )
    return records


def build_payload() -> dict[str, object]:
    events, features = load_events()
    probabilities, importance, metrics, _, _ = fit_model(events, features)
    specs = feature_specs(events, probabilities, importance)
    shown_features = list(specs)[:12]
    return {
        "threshold": THRESHOLD,
        "metrics": metrics,
        "labels": {key: LABELS.get(key, key) for key in shown_features},
        "specs": {key: specs[key] for key in shown_features},
        "events": sampled_records(events, probabilities, shown_features),
        "source": {
            "rows": int(len(events)),
            "from": events["event_at"].min().strftime("%Y-%m-%d %H:%M:%S"),
            "to": events["event_at"].max().strftime("%Y-%m-%d %H:%M:%S"),
        },
    }


def train_and_save_model() -> tuple[pd.DataFrame, np.ndarray, dict[str, object]]:
    """Train once from the historical CSV and persist the production inference artifact."""
    events, features = load_events()
    probabilities, importance, metrics, model, medians = fit_model(events, features)
    specs = feature_specs(events, probabilities, importance)
    artifact: dict[str, object] = {
        "model": model,
        "features": features,
        "medians": medians,
        "threshold": THRESHOLD,
        "metrics": metrics,
        "labels": LABELS,
        "specs": specs,
    }
    MODEL_OUT.parent.mkdir(exist_ok=True)
    MODEL_OUT.write_bytes(pickle.dumps(artifact))
    return events, probabilities, artifact


TEMPLATE = r'''<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
  <title>Foundry Guard | AI 공정 관제</title>
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
  <style>
    :root{--bg:#f5f7fb;--panel:#fff;--ink:#14213d;--muted:#64748b;--line:#e2e8f0;--blue:#2563eb;--green:#16a34a;--amber:#d97706;--red:#dc2626;--nav:#0f172a}*{box-sizing:border-box}body{margin:0;font:14px/1.45 Inter,ui-sans-serif,system-ui,sans-serif;color:var(--ink);background:var(--bg)}button,input,select{font:inherit}.layout{display:grid;grid-template-columns:248px 1fr;min-height:100vh}.side{background:var(--nav);color:#cbd5e1;padding:24px 14px;position:sticky;top:0;height:100vh}.brand{display:flex;gap:10px;align-items:center;padding:0 10px 26px;color:#fff;font-weight:800;font-size:17px}.logo{display:grid;place-items:center;width:30px;height:30px;background:#2563eb;border-radius:9px}.side small{color:#94a3b8}.nav{display:grid;gap:4px;margin-top:20px}.nav button{border:0;background:transparent;color:#cbd5e1;text-align:left;border-radius:8px;padding:11px 12px;cursor:pointer}.nav button:hover,.nav button.active{background:#1e293b;color:#fff}.state{position:absolute;bottom:20px;left:24px;right:24px;border-top:1px solid #334155;padding-top:16px;font-size:12px}.dot{display:inline-block;width:8px;height:8px;border-radius:50%;background:#22c55e;margin-right:6px}.main{min-width:0}.top{background:#fff;border-bottom:1px solid var(--line);padding:17px 28px;display:flex;justify-content:space-between;gap:18px;align-items:center;position:sticky;top:0;z-index:4}.filters{display:flex;gap:8px;flex-wrap:wrap;align-items:center}.filters input,.filters select{border:1px solid var(--line);padding:7px 9px;border-radius:7px;background:#fff}.content{padding:25px 28px;max-width:1700px}.heading{display:flex;justify-content:space-between;gap:16px;align-items:end;margin-bottom:18px}.heading h1{font-size:23px;margin:0}.heading p{color:var(--muted);margin:4px 0 0}.view{display:none}.view.active{display:block}.kpis{display:grid;grid-template-columns:repeat(7,minmax(130px,1fr));gap:12px}.card{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:16px;box-shadow:0 1px 2px #0f172a08}.kpi .name{font-size:12px;color:var(--muted)}.kpi .value{font-size:24px;font-weight:750;margin:6px 0 2px}.kpi .note{font-size:11px;color:var(--muted)}.grid{display:grid;grid-template-columns:repeat(12,minmax(0,1fr));gap:14px;margin-top:14px}.span-12{grid-column:span 12}.span-8{grid-column:span 8}.span-7{grid-column:span 7}.span-6{grid-column:span 6}.span-5{grid-column:span 5}.chart{height:320px}.card h2{font-size:14px;margin:0 0 10px}.card .sub{font-size:12px;color:var(--muted);margin:-5px 0 8px}.status{border-radius:10px;padding:15px;background:#fff7ed;border:1px solid #fed7aa}.status.danger{background:#fef2f2;border-color:#fecaca}.status.ok{background:#f0fdf4;border-color:#bbf7d0}.status h2{margin:0 0 4px}.status .reason{color:#475569;font-size:13px}.next{margin-top:10px;padding:10px;background:#fff;border-radius:8px;font-size:12px}.table-wrap{overflow:auto}.table{width:100%;border-collapse:collapse;font-size:12px}.table th{color:#64748b;font-weight:600;text-align:left;background:#f8fafc}.table td,.table th{padding:10px;border-bottom:1px solid var(--line);white-space:nowrap}.badge{font-size:11px;border-radius:999px;padding:3px 7px;font-weight:700}.badge.a,.badge.critical{background:#fee2e2;color:#b91c1c}.badge.b,.badge.warning{background:#fef3c7;color:#92400e}.badge.c,.badge.normal{background:#dcfce7;color:#166534}.muted{color:var(--muted)}.warning-row{background:#fff7ed}.critical-row{background:#fef2f2}.spec-bar{height:8px;border-radius:99px;background:#e2e8f0;position:relative;min-width:120px}.spec-bar span{position:absolute;top:0;height:8px;background:#22c55e;border-radius:99px}.spec-bar i{position:absolute;top:-4px;width:2px;height:16px;background:#0f172a}.metric{display:flex;align-items:center;gap:8px}.tiny{font-size:11px;color:var(--muted)}@media(max-width:1200px){.kpis{grid-template-columns:repeat(3,1fr)}.span-8,.span-7,.span-6,.span-5{grid-column:span 12}}@media(max-width:760px){.layout{display:block}.side{display:none}.top{padding:13px}.content{padding:16px}.kpis{grid-template-columns:repeat(2,1fr)}.heading{display:block}.chart{height:280px}}
  </style>
</head>
<body>
<div class="layout"><aside class="side"><div class="brand"><span class="logo">◈</span>Foundry Guard</div><small>AI 품질 예방 관제</small><div class="nav"><button class="active" data-view="overview">통합 현황</button><button data-view="process">공정 분석</button><button data-view="specs">스펙 센터</button><button data-view="alerts">알림·조치</button></div><div class="state"><span class="dot"></span>데모 데이터 연결<br><small id="sourceRange"></small></div></aside>
<main class="main"><header class="top"><div><b>Line 01</b> <span class="muted">/ CylinderHead</span></div><div class="filters"><label>기간 <input id="from" type="date"></label><span>~</span><input id="to" type="date"><select id="risk"><option value="all">전체 위험</option><option value="high">고위험만</option><option value="defect">실제 불량만</option></select><button id="reset">초기화</button></div></header>
<section class="content">
  <div class="view active" id="overview"><div class="heading"><div><h1>통합 현황</h1><p>불량을 예측하고, 스펙·공정변동을 함께 감시합니다.</p></div><span class="tiny" id="updated"></span></div><div class="kpis" id="kpis"></div><div class="grid"><article class="card span-8"><h2>품질 트렌드</h2><p class="sub">실측 불량률과 고위험 판정률</p><div id="trend" class="chart"></div></article><article class="status span-4" id="guard"></article><article class="card span-5"><h2>최근 고위험 이벤트</h2><div class="table-wrap"><table class="table"><thead><tr><th>시각</th><th>위험</th><th>판정</th><th>우선 확인</th></tr></thead><tbody id="recent"></tbody></table></div></article><article class="card span-7"><h2>핵심 피쳐 상태</h2><p class="sub">AI 운영구간은 후보이며 현업 승인 전입니다.</p><div class="table-wrap"><table class="table"><thead><tr><th>항목</th><th>현재값</th><th>후보 구간</th><th>상태</th></tr></thead><tbody id="featureState"></tbody></table></div></article></div></div>
  <div class="view" id="process"><div class="heading"><div><h1>공정 분석</h1><p>시계열, 양·불량 분포, 관계 탐색</p></div><select id="featureSelect"></select></div><div class="grid"><article class="card span-12"><h2 id="featureTitle">공정 시계열</h2><p class="sub">AI 후보 구간은 녹색 영역, 최근값은 검은 점으로 표시</p><div id="processTrend" class="chart"></div></article><article class="card span-6"><h2>양·불량 box plot</h2><div id="box" class="chart"></div></article><article class="card span-6"><h2>산점도: 주입 압력 vs 하부 온도 2</h2><div id="scatter" class="chart"></div></article></div></div>
  <div class="view" id="specs"><div class="heading"><div><h1>스펙 센터</h1><p>현업 규격, AI 운영구간, SPC는 서로 다른 종류의 기준입니다.</p></div></div><div class="card"><h2>피쳐 관리 등급 및 AI 운영구간 후보</h2><p class="sub">현 데모에서는 저위험 생산 이벤트의 5~95백분위 범위로 후보를 제시합니다. 승인 규격이 아닙니다.</p><div class="table-wrap"><table class="table"><thead><tr><th>등급</th><th>피쳐</th><th>모델 중요도</th><th>AI 후보 운영구간</th><th>최근값</th><th>현재 상태</th><th>운영 단계</th></tr></thead><tbody id="specTable"></tbody></table></div></div></div>
  <div class="view" id="alerts"><div class="heading"><div><h1>알림·조치</h1><p>예측·스펙·변동을 결합한 예방 경보 목록</p></div></div><div class="card"><h2>활성 경보 후보</h2><p class="sub">MVP는 조치 기록을 브라우저 세션에 보관합니다.</p><div class="table-wrap"><table class="table"><thead><tr><th>중요도</th><th>시각</th><th>경보</th><th>근거</th><th>조치</th></tr></thead><tbody id="alertsTable"></tbody></table></div></div></div>
</section></main></div>
<script>const APP=__PAYLOAD__;
const $=s=>document.querySelector(s), fmt=n=>Number(n).toLocaleString('ko-KR',{maximumFractionDigits:2}), pct=n=>(n*100).toFixed(1)+'%', layout={margin:{l:46,r:20,t:14,b:40},paper_bgcolor:'white',plot_bgcolor:'white',font:{color:'#334155'},xaxis:{gridcolor:'#edf2f7'},yaxis:{gridcolor:'#edf2f7'}};
const features=Object.keys(APP.specs), label=f=>APP.labels[f]||f;
function filtered(){const a=$('#from').value,b=$('#to').value,r=$('#risk').value;return APP.events.filter(e=>(!a||e.d>=a)&&(!b||e.d<=b)&&(r==='all'||r==='high'&&e.p>=APP.threshold||r==='defect'&&e.y===1));}
function daily(rows){const groups={};rows.forEach(e=>{const g=groups[e.d]??(groups[e.d]={n:0,bad:0,high:0,p:0});g.n++;g.bad+=e.y;g.high+=e.p>=APP.threshold;g.p+=e.p});return Object.entries(groups).map(([d,g])=>({d,...g,bad:g.bad/g.n,high:g.high/g.n,p:g.p/g.n}));}
function current(rows,f){return [...rows].reverse().find(e=>e.v[f]!=null)?.v[f]}
function outside(v,s){return v!=null&&(v<s.low||v>s.high)}
function priority(e){const breaches=features.filter(f=>outside(e.v[f],APP.specs[f]));return {breaches,critical:e.p>=APP.threshold&&breaches.some(f=>APP.specs[f].grade==='A'),warning:e.p>=APP.threshold||breaches.length>0}}
function renderOverview(rows){const high=rows.filter(e=>e.p>=APP.threshold).length, defects=rows.filter(e=>e.y).length, recent=rows.at(-1), alertCount=rows.filter(e=>priority(e).warning).length;const cards=[['생산 수량',fmt(rows.length),'필터 구간 이벤트'],['실측 불량률',pct(defects/Math.max(rows.length,1)),fmt(defects)+' / '+fmt(rows.length)],['평균 예측위험',pct(rows.reduce((s,e)=>s+e.p,0)/Math.max(rows.length,1)),'모델 확률 평균'],['고위험 판정률',pct(high/Math.max(rows.length,1)),APP.threshold+'이상 '+fmt(high)+'건'],['핵심 스펙 이탈',fmt(features.filter(f=>outside(recent?.v[f],APP.specs[f])&&APP.specs[f].grade==='A').length,'현재 이벤트 기준'],['공정 변동 이상',fmt(features.filter(f=>{const s=APP.specs[f],v=current(rows,f);return v!=null&&Math.abs((v-s.mean)/Math.max(s.std,.001))>3}).length,'3σ 기준'],['활성 경보',fmt(alertCount),'예방 경보 후보']];$('#kpis').innerHTML=cards.map(c=>`<article class="card kpi"><div class="name">${c[0]}</div><div class="value">${c[1]}</div><div class="note">${c[2]}</div></article>`).join('');const d=daily(rows);Plotly.react('trend',[{x:d.map(x=>x.d),y:d.map(x=>x.bad*100),name:'실측 불량률',type:'scatter',line:{color:'#dc2626',width:2}},{x:d.map(x=>x.d),y:d.map(x=>x.high*100),name:'고위험 판정률',type:'scatter',line:{color:'#2563eb',width:2,dash:'dot'}}],{...layout,yaxis:{...layout.yaxis,ticksuffix:'%'},legend:{orientation:'h'}});const p=priority(recent||{}), cls=p.critical?'danger':p.warning?'':'ok', title=p.critical?'위험: 즉시 확인':p.warning?'주의: 선제 점검':'정상: 현재 이상 없음', reasons=[recent?.p>=APP.threshold?`예측 불량 위험 ${pct(recent.p)}`:null,...p.breaches.map(f=>`${label(f)} 후보구간 이탈`)].filter(Boolean);$('#guard').className='status span-4 '+cls;$('#guard').innerHTML=`<h2>${title}</h2><div class="reason">${reasons.join(' · ')||'스펙 후보와 위험 임계값에서 안정적입니다.'}</div><div class="next"><b>다음 조치</b><br>${p.breaches.slice(0,3).map(f=>label(f)+' 실제값·센서·SOP 확인').join('<br>')||'최근 트렌드를 유지 관찰하세요.'}</div>`;const highRows=[...rows].reverse().filter(e=>e.p>=APP.threshold).slice(0,6);$('#recent').innerHTML=highRows.map(e=>{const q=priority(e);return `<tr><td>${e.t.slice(5)}</td><td>${pct(e.p)}</td><td>${e.y?'<span class="badge a">불량</span>':'<span class="badge c">미확정</span>'}</td><td>${q.breaches.slice(0,2).map(label).join(', ')||'예측 위험'}</td></tr>`}).join('')||'<tr><td colspan="4" class="muted">고위험 이벤트 없음</td></tr>';$('#featureState').innerHTML=features.slice(0,7).map(f=>{const s=APP.specs[f],v=current(rows,f),bad=outside(v,s);return `<tr><td><span class="badge ${s.grade.toLowerCase()}">${s.grade}</span> ${label(f)}</td><td>${fmt(v)}</td><td>${fmt(s.low)} ~ ${fmt(s.high)}</td><td>${bad?'<span class="badge warning">주의</span>':'<span class="badge normal">정상</span>'}</td></tr>`).join('')}
function renderProcess(rows){const f=$('#featureSelect').value||features[0],s=APP.specs[f],x=rows.map(e=>e.t),y=rows.map(e=>e.v[f]);$('#featureTitle').textContent=label(f)+' 공정 시계열';Plotly.react('processTrend',[{x,y,type:'scattergl',mode:'lines',line:{color:'#2563eb'},name:label(f)},{x:[rows.at(-1)?.t],y:[current(rows,f)],mode:'markers',marker:{color:'#0f172a',size:9},name:'최근값'}],{...layout,shapes:[{type:'rect',xref:'paper',x0:0,x1:1,y0:s.low,y1:s.high,fillcolor:'rgba(34,197,94,.15)',line:{width:0}}],legend:{orientation:'h'}});const good=rows.filter(e=>!e.y).map(e=>e.v[f]),bad=rows.filter(e=>e.y).map(e=>e.v[f]);Plotly.react('box',[{y:good,type:'box',name:'양품',marker:{color:'#16a34a'}},{y:bad,type:'box',name:'불량',marker:{color:'#dc2626'}}],layout);const sx='injection_pressure',sy='bottom_temp2';const a=rows.filter(e=>!e.y),b=rows.filter(e=>e.y);Plotly.react('scatter',[{x:a.map(e=>e.v[sx]),y:a.map(e=>e.v[sy]),mode:'markers',type:'scattergl',name:'양품',marker:{color:'rgba(22,163,74,.35)',size:5}},{x:b.map(e=>e.v[sx]),y:b.map(e=>e.v[sy]),mode:'markers',type:'scattergl',name:'불량',marker:{color:'rgba(220,38,38,.55)',size:5}}],{...layout,xaxis:{title:label(sx)},yaxis:{title:label(sy)},legend:{orientation:'h'}})}
function renderSpecs(rows){const last=rows.at(-1);$('#specTable').innerHTML=features.map(f=>{const s=APP.specs[f],v=current(rows,f),bad=outside(v,s),range=Math.max(s.high-s.low,.001),pos=Math.min(100,Math.max(0,(v-s.low)/range*100));return `<tr class="${bad?'warning-row':''}"><td><span class="badge ${s.grade.toLowerCase()}">${s.grade}</span></td><td>${label(f)}</td><td>${pct(s.importance)}</td><td><div class="metric">${fmt(s.low)} ~ ${fmt(s.high)}<div class="spec-bar"><span style="left:0;width:100%"></span><i style="left:${pos}%"></i></div></div></td><td>${fmt(v)}</td><td>${bad?'<span class="badge warning">후보 이탈</span>':'<span class="badge normal">후보 내</span>'}</td><td><span class="tiny">Shadow 후보</span></td></tr>`}).join('')}
function renderAlerts(rows){const alerts=[...rows].reverse().flatMap(e=>{const q=priority(e);if(!q.warning)return [];const names=q.breaches.map(label);const critical=q.critical;return [{e,critical,names}] }).slice(0,40);$('#alertsTable').innerHTML=alerts.map(({e,critical,names},i)=>`<tr class="${critical?'critical-row':'warning-row'}"><td><span class="badge ${critical?'critical':'warning'}">${critical?'Critical':'Warning'}</span></td><td>${e.t}</td><td>${critical?'복합 예방 경보':'고위험 또는 스펙 후보 이탈'}</td><td>예측 ${pct(e.p)}${names.length?' · '+names.join(', '):''}</td><td><button data-action="${i}">확인 및 조치 기록</button></td></tr>`).join('')||'<tr><td colspan="5" class="muted">필터 기준 경보 후보 없음</td></tr>';document.querySelectorAll('[data-action]').forEach(b=>b.onclick=()=>{b.textContent='확인됨';b.disabled=true;localStorage.setItem('foundry-action-'+b.dataset.action,new Date().toISOString())})}
function render(){const rows=filtered();renderOverview(rows);if(!rows.length){$('#recent').innerHTML='';$('#featureState').innerHTML='';$('#specTable').innerHTML='<tr><td colspan="7" class="muted">해당 기간 데이터 없음</td></tr>';$('#alertsTable').innerHTML='<tr><td colspan="5" class="muted">해당 기간 경보 없음</td></tr>';$('#updated').textContent='표시 데이터 0건';return}renderProcess(rows);renderSpecs(rows);renderAlerts(rows);$('#updated').textContent='표시 데이터 '+fmt(rows.length)+'건 · 임계값 '+APP.threshold}
document.querySelectorAll('[data-view]').forEach(b=>b.onclick=()=>{document.querySelectorAll('[data-view]').forEach(x=>x.classList.remove('active'));document.querySelectorAll('.view').forEach(x=>x.classList.remove('active'));b.classList.add('active');$('#'+b.dataset.view).classList.add('active');setTimeout(()=>render(),0)});
const dates=APP.events.map(e=>e.d);$('#from').value=Math.min(...dates.map(Date.parse))?dates[0]:'';$('#to').value=dates.at(-1);features.forEach(f=>$('#featureSelect').insertAdjacentHTML('beforeend',`<option value="${f}">${label(f)}</option>`));['from','to','risk','featureSelect'].forEach(id=>$('#'+id).onchange=render);$('#reset').onclick=()=>{$('#from').value=dates[0];$('#to').value=dates.at(-1);$('#risk').value='all';render()};$('#sourceRange').textContent=APP.source.from+' ~ '+APP.source.to;render();
</script></body></html>'''

OVERVIEW_SCRIPT = r'''function renderOverview(rows){
  if(!rows.length){$('#kpis').innerHTML='<article class="card">\ud574당 기간 데이터 없음</article>';return}
  const high=rows.filter(e=>e.p>=APP.threshold).length,defects=rows.filter(e=>e.y).length,recent=rows.at(-1),alertCount=rows.filter(e=>priority(e).warning).length;
  const core=features.filter(f=>outside(recent.v[f],APP.specs[f])&&APP.specs[f].grade==='A').length;
  const drift=features.filter(f=>{const s=APP.specs[f],v=current(rows,f);return v!=null&&Math.abs((v-s.mean)/Math.max(s.std,.001))>3}).length;
  const cards=[['생산 수량',fmt(rows.length),'필터 구간 이벤트'],['실측 불량률',pct(defects/rows.length),fmt(defects)+' / '+fmt(rows.length)],['평균 예측위험',pct(rows.reduce((sum,e)=>sum+e.p,0)/rows.length),'모델 확률 평균'],['고위험 판정률',pct(high/rows.length),APP.threshold+'이상 '+fmt(high)+'건'],['핵심 스펙 이탈',fmt(core),'현재 이벤트 기준'],['공정 변동 이상',fmt(drift),'3σ 기준'],['활성 경보',fmt(alertCount),'예방 경보 후보']];
  $('#kpis').innerHTML=cards.map(c=>'<article class="card kpi"><div class="name">'+c[0]+'</div><div class="value">'+c[1]+'</div><div class="note">'+c[2]+'</div></article>').join('');
  const d=daily(rows);Plotly.react('trend',[{x:d.map(x=>x.d),y:d.map(x=>x.bad*100),name:'실측 불량률',type:'scatter',line:{color:'#dc2626',width:2}},{x:d.map(x=>x.d),y:d.map(x=>x.high*100),name:'고위험 판정률',type:'scatter',line:{color:'#2563eb',width:2,dash:'dot'}}],{...layout,yaxis:{...layout.yaxis,ticksuffix:'%'},legend:{orientation:'h'}});
  const p=priority(recent),cls=p.critical?'danger':p.warning?'':'ok',title=p.critical?'위험: 즉시 확인':p.warning?'주의: 선제 점검':'정상: 현재 이상 없음',reasons=[recent.p>=APP.threshold?'예측 불량 위험 '+pct(recent.p):null,...p.breaches.map(f=>label(f)+' 후보구간 이탈')].filter(Boolean);
  $('#guard').className='status span-4 '+cls;$('#guard').innerHTML='<h2>'+title+'</h2><div class="reason">'+(reasons.join(' · ')||'스펙 후보와 위험 임계값에서 안정적입니다.')+'</div><div class="next"><b>다음 조치</b><br>'+(p.breaches.slice(0,3).map(f=>label(f)+' 실제값·센서·SOP 확인').join('<br>')||'최근 트렌드를 유지 관찰하세요.')+'</div>';
  const highRows=[...rows].reverse().filter(e=>e.p>=APP.threshold).slice(0,6);$('#recent').innerHTML=highRows.map(e=>{const q=priority(e);return '<tr><td>'+e.t.slice(5)+'</td><td>'+pct(e.p)+'</td><td>'+(e.y?'<span class="badge a">불량</span>':'<span class="badge c">미확정</span>')+'</td><td>'+(q.breaches.slice(0,2).map(label).join(', ')||'예측 위험')+'</td></tr>'}).join('')||'<tr><td colspan="4" class="muted">고위험 이벤트 없음</td></tr>';
  $('#featureState').innerHTML=features.slice(0,7).map(f=>{const s=APP.specs[f],v=current(rows,f),bad=outside(v,s);return '<tr><td><span class="badge '+s.grade.toLowerCase()+'">'+s.grade+'</span> '+label(f)+'</td><td>'+fmt(v)+'</td><td>'+fmt(s.low)+' ~ '+fmt(s.high)+'</td><td>'+(bad?'<span class="badge warning">주의</span>':'<span class="badge normal">정상</span>')+'</td></tr>'}).join('');
}
'''


def build() -> Path:
    payload = build_payload()
    start = TEMPLATE.index("function renderOverview")
    end = TEMPLATE.index("function renderProcess")
    html = TEMPLATE[:start] + OVERVIEW_SCRIPT + TEMPLATE[end:]
    OUT.write_text(html.replace("__PAYLOAD__", json.dumps(payload, ensure_ascii=False, separators=(",", ":"))), encoding="utf-8")
    return OUT


if __name__ == "__main__":
    output = build()
    assert output.exists() and output.stat().st_size > 100_000
    print(f"Wrote {output.name} ({output.stat().st_size:,} bytes)")
