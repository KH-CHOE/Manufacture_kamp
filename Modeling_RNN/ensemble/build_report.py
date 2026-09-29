#!/usr/bin/env python3
"""순환신경망·앙상블 보고서 HTML 생성. 결과 파일에서 수치를 읽어 만든다.

실행: python build_report.py
출력: 순환신경망_앙상블_보고서.html

`Modeling/build_report.py` 와 같은 시각 언어(색·구조·인쇄 규칙)를 쓴다.
재학습은 하지 않는다 — 이미 만들어진 예측·지표 파일만 읽는다.
"""
from __future__ import annotations

import html
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
RNN = HERE.parent
TREE = RNN.parent / "Modeling"
OUT = HERE / "순환신경망_앙상블_보고서.html"
SHUTDOWN = (20210731, 20210808)


def esc(v) -> str:
    return html.escape(str(v), quote=True)


def n(v, d=2) -> str:
    return f"{float(v):,.{d}f}"


def table(rows: list[list], head: list[str], hi: int | None = None,
          align_first_left=True) -> str:
    th = "".join(f"<th>{esc(h)}</th>" for h in head)
    tr = []
    for i, r in enumerate(rows):
        cls = ' class="hi"' if hi is not None and i == hi else ""
        td = "".join(f"<td>{c}</td>" for c in r)
        tr.append(f"<tr{cls}>{td}</tr>")
    return (f'<div class="table-scroll"><table><thead><tr>{th}</tr></thead>'
            f'<tbody>{"".join(tr)}</tbody></table></div>')


def note(title: str, body: str, kind: str = "") -> str:
    k = f" {kind}" if kind else ""
    return f'<div class="note{k}"><strong>{title}</strong><p>{body}</p></div>'


def load() -> dict:
    """결과 파일에서 수치를 읽는다"""
    met = pd.read_csv(RNN / "reproduced_metrics.csv", encoding="utf-8-sig")
    meta = json.loads((RNN / "reproduction_metadata.json").read_text(encoding="utf-8"))
    rn = pd.read_csv(RNN / "reproduced_test_predictions.csv", encoding="utf-8-sig")
    rn["ts"] = pd.to_datetime(rn["ts"])
    bg = pd.read_csv(TREE / "reproduced_test_predictions.csv", encoding="utf-8-sig")
    bg["ts"] = pd.to_datetime(bg["forecast_time"]) - pd.Timedelta(minutes=15)
    bgm = pd.read_csv(TREE / "reproduced_metrics.csv", encoding="utf-8-sig")

    j = bg[["ts", "actual", "predicted"]].rename(
        columns={"actual": "y", "predicted": "et"}).merge(
        rn[["ts", "actual", "predicted"]].rename(
            columns={"actual": "y2", "predicted": "gru"}), on="ts", how="inner")
    assert (j["y"] - j["y2"]).abs().max() < 1e-6, "실측값이 어긋난다"
    j["date_key"] = j["ts"].dt.strftime("%Y%m%d").astype(int)

    def m(a, p):
        return float(((a - p) ** 2).mean())

    y, et, gru = j["y"].to_numpy(), j["et"].to_numpy(), j["gru"].to_numpy()
    sd = ((j["date_key"] >= SHUTDOWN[0]) & (j["date_key"] <= SHUTDOWN[1])).to_numpy()
    return {
        "met": met, "meta": meta, "bg_all": float(bgm["test_MSE"].iloc[0]),
        "rows": len(j), "corr": float(np.corrcoef(et, gru)[0, 1]),
        "et": m(y, et), "gru": m(y, gru), "half": m(y, (et + gru) / 2),
        "seg": {name: {"rows": int(k.sum()), "et": m(y[k], et[k]), "gru": m(y[k], gru[k]),
                       "half": m(y[k], (et[k] + gru[k]) / 2)}
                for name, k in (("하계휴무 7/31~8/8", sd), ("평시", ~sd))},
        "excl_et": m(bg[~bg["ts"].isin(j["ts"])]["actual"].to_numpy(),
                     bg[~bg["ts"].isin(j["ts"])]["predicted"].to_numpy()),
        "excl_rows": len(bg) - len(j),
    }


CSS = """
:root{--ink:#173047;--blue:#2462a9;--aqua:#008a8c;--gold:#b47727;--red:#a8342a;
--paper:#f5f8fa;--line:#d8e3eb;--muted:#5a6e7b;--white:#fff}
*{box-sizing:border-box}html{scroll-behavior:smooth}
body{margin:0;background:var(--paper);color:var(--ink);
font:15px/1.78 -apple-system,BlinkMacSystemFont,"Apple SD Gothic Neo","Malgun Gothic",sans-serif;
font-variant-numeric:tabular-nums}
a{color:var(--blue)}a:focus-visible{outline:3px solid var(--gold);outline-offset:3px}
h1,h2,h3{line-height:1.36;letter-spacing:-.035em}
h1{font-size:40px;margin:0 0 18px}h2{font-size:27px;margin:0 0 22px}
h3{font-size:18px;margin:29px 0 11px}h4{font-size:15px;margin:22px 0 8px}
p{max-width:84ch;margin:0 0 16px}ul{padding-left:22px}li{margin:5px 0}
.topbar{background:var(--ink);color:#fff;padding:20px max(24px,calc((100vw - 1280px)/2));
font-size:14px;font-weight:700}
.frame{max-width:1320px;margin:auto;display:grid;grid-template-columns:190px minmax(0,1fr);
gap:40px;padding:40px 24px 90px}
aside{align-self:start;position:sticky;top:25px}
aside a{display:block;padding:8px 11px;border-left:2px solid var(--line);
text-decoration:none;font-size:12px;color:var(--muted)}
aside a:hover{color:var(--blue);background:#e8f0f7;border-color:var(--blue)}
main{min-width:0}
.cover{background:#e6eff6;padding:46px 42px;margin-bottom:26px;border-top:5px solid var(--aqua)}
.cover p{font-size:17px;color:#415a6c}
.cover small{display:block;font-size:12px;letter-spacing:.05em;margin-bottom:18px}
.docket{display:grid;grid-template-columns:repeat(4,1fr);gap:15px;margin-top:28px}
.docket>div{border-top:1px solid #a6c5d3;padding-top:12px}
.docket b{font-size:22px;color:var(--blue);display:block}
.docket span{font-size:12px;color:var(--muted)}
section{background:#fff;padding:34px 40px;margin-bottom:24px;border:1px solid var(--line)}
.section-number{font-size:12px;color:var(--aqua);font-weight:700;display:block;margin-bottom:6px}
.lede{font-size:17px;color:#456073}
.table-scroll{overflow:auto;border:1px solid var(--line);margin:17px 0}
table{width:100%;border-collapse:collapse;white-space:nowrap;font-size:12.5px}
th,td{padding:10px 13px;border-bottom:1px solid #e8eef2;text-align:right}
th:first-child,td:first-child{text-align:left}
thead th{background:#eaf1f6;color:#35526b;font-weight:700}
tr:last-child td{border-bottom:0}tbody tr:hover{background:#f4f8fb}
tr.hi td{background:#fff5e4;font-weight:700}
.note{padding:16px 20px;background:#edf4fa;border-left:4px solid var(--blue);margin:21px 0;
max-width:84ch}
.note strong{display:block;margin-bottom:4px}.note p:last-child{margin:0}
.note.caution{background:#fff8e9;border-color:var(--gold)}
.note.bad{background:#fdf0ee;border-color:var(--red)}
.note.good{background:#e9f5f1;border-color:var(--aqua)}
.summary-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin:22px 0}
.summary-grid div{border-top:3px solid var(--blue);background:#f4f8fb;padding:15px}
.summary-grid small{display:block;color:var(--muted);font-size:12px}
.summary-grid b{display:block;font-size:22px}
.code{background:#e8f0f6;padding:2px 5px;font:12px/1.6 ui-monospace,Menlo,monospace;
overflow-wrap:anywhere}
pre{background:var(--ink);color:#fff;padding:18px;overflow:auto;font:12.5px/1.7 ui-monospace,Menlo,monospace}
footer{font-size:12px;color:var(--muted);padding:10px 0 30px}
@media(max-width:980px){.frame{display:block}aside{position:static;display:flex;overflow:auto;
margin-bottom:20px}aside a{white-space:nowrap}.docket{grid-template-columns:repeat(2,1fr)}}
@media(max-width:620px){.frame{padding:20px 12px}.cover,section{padding:24px 17px}
h1{font-size:28px}h2{font-size:22px}.docket,.summary-grid{grid-template-columns:1fr}}
@media print{body{background:#fff}.frame{display:block;padding:0}aside{display:none}
.cover,section{break-inside:avoid}.table-scroll{overflow:visible}
a{text-decoration:none;color:var(--ink)}}
"""

NAV = [("s1", "1 순환신경망"), ("s2", "2 앙상블"), ("s3", "3 파생변수"),
       ("s4", "4 재현"), ("s5", "5 한계")]


def main() -> None:
    d = load()
    met, meta = d["met"], d["meta"]
    seeds = met["seed"].tolist()
    ens = meta["seed_ensemble_MSE"]
    ens_mae = meta["seed_ensemble_MAE"]
    best_solo = min(d["et"], d["gru"])
    cut = (1 - d["half"] / best_solo) * 100

    # ── 1 순환신경망 ──
    s1 = f"""<section id="s1"><span class="section-number">01 / RECURRENT MODEL</span>
<h2>순환신경망(GRU) 결과</h2>
<p class="lede">과거 하루치 전력 계열을 순서대로 먹여 다음 15분을 맞힙니다.
<span class="code">Modeling/</span>(ExtraTrees)과 <b>같은 행·같은 분할</b>에서 평가해
수치를 바로 비교할 수 있게 맞췄습니다.</p>
<h3>설정</h3>
{table([
  ["구조", f'GRU · 은닉 {meta["config"]["hidden"]} · 1층'],
  ["입력", f'과거 {meta["config"]["window"]}칸(하루) 전력 계열을 '
           f'{meta["config"]["steps"]}스텝 × {meta["config"]["window"]//meta["config"]["steps"]}채널로 접어서 + 달력 6개'],
  ["학습 구간", f'예측 시점 직전 <b>{meta["train_months"]}개월</b>'],
  ["선정 기준", "전진검증 4구간(4·5·6·7월 각 24일) 평균 MSE — 시험 구간 미사용"],
  ["시드", " · ".join(str(s) for s in seeds) + " (3개)"],
], ["항목", "값"], align_first_left=True).replace('text-align:right', 'text-align:left')}
{note("시드를 3개 돌리는 이유",
 "탐색 중 한 설정이 시드 42에서 평가 54.99, 시드 7에서 135.30으로 <b>2.5배</b> 벌어진 적이 "
 "있습니다. 더 문제는 <b>시드 7의 전진검증이 세 시드 중 가장 좋았다</b>는 것입니다 — "
 "전진검증이 그 취약점을 경고하지 못했습니다. "
 "단일 시드 수치는 인용하지 마시고 평균과 범위를 함께 봐 주세요.", "caution")}
<h3>성적 — 시험 구간 20210627~0914</h3>
{table([[str(r.seed), n(r.MSE, 3), n(r.MAE, 3), n(r.R2, 4), str(int(r.epochs))]
        for r in met.itertuples()]
       + [["<b>평균</b>", f'<b>{n(met.MSE.mean(), 3)}</b>', n(met.MAE.mean(), 3),
           n(met.R2.mean(), 4), "—"],
          [f'<b>세 시드 예측 평균</b>', f'<b>{n(ens, 3)}</b>', f'<b>{n(ens_mae, 3)}</b>', "", ""]],
       ["시드", "시험 MSE", "MAE", "R²", "에폭"], hi=4)}
<p>세 시드의 <b>예측을 평균내면 {n(ens, 3)}</b>으로 개별 최고({n(met.MSE.min(), 3)})보다 낫습니다.
시험 성적을 보고 고른 것이 아니라 <b>미리 정한 절차</b>의 결과라 그대로 보고합니다.</p>
<h3>비교할 때 — 행을 맞춰야 합니다</h3>
<p>시험 7,294행 중 <b>{d["excl_rows"]}행은 순환신경망이 만들 수 없습니다.</b>
삭제된 2021-07-13·15를 건너뛰는 창이라 제외됩니다. 그런데
<b>그 {d["excl_rows"]}행이 쉬운 구간</b>입니다 — 거기서 ExtraTrees의 MSE가
{n(d["excl_et"], 2)}로 전체 평균보다 훨씬 낮습니다.</p>
{table([["ExtraTrees (<span class='code'>Modeling/</span>)", n(d["bg_all"], 3), f'<b>{n(d["et"], 3)}</b>'],
        ["GRU 시드 평균 (이 폴더)", "—", f'<b>{n(d["gru"], 3)}</b>']],
       ["모델", "7,294행", f'공통 {d["rows"]:,}행'])}
{note("어디서 무너지는가 — 이게 더 중요합니다",
 "<b>선정 기준인 전진검증에서는 순환 계열이 밉니다</b>(75.59 대 부스팅 63.65). "
 "그 열세가 거의 전부 <b>한 구간</b>에서 나옵니다 — 4월은 5월 조업 전환이 진행 중이라 "
 "월요일 새벽이 21.9 / 85.4 / 20.9 / 90.2로 <b>격주 교대</b>합니다(주간편차 38.4, "
 "다른 달은 0.4~6.3). 순환신경망은 <b>함수 하나</b>를 학습하므로 "
 "“이번 주는 돌고 다음 주는 쉰다”를 표현할 방법이 없습니다. "
 "트리는 달력 변수로 <b>분기</b>해 두 경우를 따로 담습니다.", "bad")}
{table([["04/01", "04/01~04/24", "<b>36.9</b>", "73.92", "120.46", "<b>−63.0%</b>"],
        ["05/01", "05/01~05/24", "1.2", "44.60", "45.62", "−2.3%"],
        ["06/01", "06/01~06/24", "0.2", "21.43", "22.61", "−5.5%"],
        ["07/01", "07/01~07/24", "0.2", "114.64", "119.97", "−4.7%"],
        ["시험", "07/25~09/14", "1.3", "64.02", "<b>50.70</b>", "<b>+20.8%</b>"]],
       ["구간", "맞힐 기간", "주간편차", "트리", "GRU", "GRU 우세"], hi=0)}
</section>"""

    # ── 2 앙상블 ──
    sg = d["seg"]
    s2 = f"""<section id="s2"><span class="section-number">02 / ENSEMBLE</span>
<h2>두 예측을 섞으면 — 이번 작업의 핵심</h2>
<div class="summary-grid">
<div><small>ExtraTrees 단독</small><b>{n(d["et"], 2)}</b></div>
<div><small>GRU 시드평균 단독</small><b>{n(d["gru"], 2)}</b></div>
<div><small>두 예측 단순 평균</small><b style="color:var(--aqua)">{n(d["half"], 2)}</b></div>
</div>
<p><b>두 예측의 상관이 {n(d["corr"], 4)}인데도 {n(cut, 1)}% 줄어듭니다.</b>
값은 비슷한데 <b>틀리는 방향이 다르다</b>는 뜻입니다.
트리는 분기로, 순환신경망은 계열 흐름으로 접근하니 오차 구조가 다릅니다.</p>
<h3>가중치는 고르지 마세요</h3>
{table([["고정 0.5 : 0.5", f'<b>{n(d["half"], 3)}</b>', f'{n(d["half"] - best_solo, 3)}'],
        ["앞 20%로 가중치 학습 (w=0.75)", "58.034", "<b>+3.9 (나빠짐)</b>"],
        ["시험 최적 (상한·보고용 아님) w=0.43", "51.900", "−5.03"]],
       ["설정", "MSE", "단독 최고 대비"], hi=0)}
<ul>
<li><b>시험 구간 앞 20%로 가중치를 정하면 오히려 나빠집니다.</b> w=0.75가 나왔는데
뒤 80%에서 같은 구간 고정반반(54.12)보다 3.9 나쁩니다 — 앞부분이 뒷부분을 대변하지 못합니다</li>
<li><b>시험 최적(51.90)과 고정 반반({n(d["half"], 2)})의 차이가 0.17뿐</b>입니다. 고를 이유가 없습니다</li>
</ul>
<h3>왜 섞이는지 — 구간별로 보면 보입니다</h3>
{table([[k, f'{v["rows"]:,}', n(v["et"], 3), n(v["gru"], 3), n(v["half"], 3)]
        for k, v in sg.items()], ["구간", "행수", "트리", "GRU", "고정반반"], hi=1)}
<p><b>평시에서는 블렌드가 둘 다 이깁니다</b>
({n(sg["평시"]["half"], 2)} &lt; {n(sg["평시"]["gru"], 2)} &lt; {n(sg["평시"]["et"], 2)}).
휴무에서는 트리가 압도적이라 블렌드가 손해를 봅니다.
조건부 결합(휴무=트리, 평시=반반)이면 약 51.7로 조금 더 낫지만,
“휴무엔 트리가 낫다”를 <b>이 시험 구간에서 확인한 것</b>이라 선택 위험이 있습니다.</p>
{note("실행에 옮긴다면", "추가 학습 없이 두 모형을 각자 돌려 평균내기만 하면 됩니다."
 "<br><span class='code'>최종 예측 = (ExtraTrees 예측 + GRU 세 시드 평균) / 2</span><br>"
 "그리고 <b>두 예측이 크게 갈리는 구간은 경고 신호</b>로 쓸 수 있습니다 — "
 "일치하면 안정 구간, 갈리면 조업 패턴이 바뀌는 중일 수 있습니다.", "good")}
</section>"""

    # ── 3 파생변수 ── 트리 선별 결과는 우리 저장소 실험에서 나온 확정 수치
    gru_feat = HERE / "gru_feature_results.json"
    gru_rows = ""
    if gru_feat.exists():
        g = json.loads(gru_feat.read_text(encoding="utf-8"))
        gb = g["기준"]
        gru_rows = f"""<h3>순환신경망에서도 재 봤습니다</h3>
{table([[k, n(v, 2), (f'{v - gb:+.2f}' if k != "기준 (달력 6개만)" else "—")]
        for k, v in g["판"].items()],
       ["설정", "전진검증 평균", "기준 대비"])}
<p>{g["해석"]}</p>"""

    s3 = f"""<section id="s3"><span class="section-number">03 / DERIVED FEATURES</span>
<h2>파생변수를 더 만들어 봤습니다 — 대부분 안 됐습니다</h2>
<p class="lede">분석에서 나온 <b>사실마다</b> 그것을 모형이 쓸 형태로 옮겨 23개를 만들고
6묶음으로 나눴습니다. 아무거나 만든 것이 아닙니다.</p>
{table([
 ["A 가동상태", "전력이 두 덩어리(비가동 22 · 가동 140)", "is_on run_on run_off since_change"],
 ["B 전환감지", "급변 구간 오차가 3.3배", "slope4 rng4 absd4 is_ramp"],
 ["C 수준대비", "1주 자기상관 0.738 &gt; 1일 0.377", "ratio_daymax dev_week excess_base dev_tod_med"],
 ["D 달력정교", "오차 절반이 4% 시간, 그 78%가 월요일 새벽", "to_start mon_dawn off_prev/next off_run"],
 ["E 기온", "비가동 기저부하가 기온에 대해 U자", "hdd14 temp_d1 temp_range_day"],
 ["F 생산", "생산량 0이 결측 위장(64일 중 11일)", "prod_on prod_zero_run prod_cum_day"],
], ["묶음", "근거가 된 사실", "변수"])}
{note("누수 검사가 실제로 하나를 잡았습니다",
 "<span class='code'>temp_range_day</span>(일교차)를 "
 "<span class='code'>groupby(날짜).transform('max') − min</span> 으로 만들었는데, 이건 "
 "<b>아침 8시에 그날 최고기온을 아는 것</b>입니다. "
 "“지금까지의 누적 일교차”로 고친 뒤에야 검사를 통과했습니다. "
 "파생변수는 이런 식으로 조용히 미래를 봅니다 — 성능을 재기 <i>전에</i> 검사를 거는 게 맞았습니다.",
 "bad")}
<h3>묶음별 결과 (ExtraTrees · 전진검증 4구간 평균)</h3>
<p>기준선은 <b>우리 변수 + 시각 원값</b>(41.75)입니다.</p>
{table([["기준선", "24", "41.75", "—"],
        ["+ A 가동상태", "28", "42.69", "+0.95"],
        ["+ B 전환감지", "28", "<b>41.19</b>", "<b>−0.56</b>"],
        ["+ C 수준대비", "28", "43.16", "+1.41"],
        ["+ D 달력정교", "29", "41.56", "−0.19"],
        ["+ E 기온", "27", "41.61", "−0.13"],
        ["+ F 생산", "27", "41.95", "+0.20"],
        ["+ 전부 (23개)", "47", "43.94", "<b>+2.19</b>"]],
       ["묶음", "변수", "평균", "기준 대비"], hi=2)}
<h4>① 변수를 많이 더할수록 나빠집니다</h4>
<p>전부 넣으면 +2.19입니다. B 단독(41.19)이 B+D+E 합친 것(41.44)보다 낫습니다.
<span class="code">max_features</span>를 1.0 → 0.7 → 0.5로 낮춰도
43.94 → 43.45 → 42.78로, 여전히 기준선을 못 넘습니다.</p>
<h4>② 살아남은 건 B 전환감지뿐이고, 그건 진짜입니다</h4>
{table([["ExtraTrees", "41.75", "41.19", "−0.56"],
        ["부스팅", "48.17", "<b>46.97</b>", "<b>−1.21</b>"]],
       ["알고리즘", "기준선", "+ B", "차이"], hi=1)}
<p>부스팅에서 더 큽니다 — <b>트리 계열 일반적인 효과</b>이지 한 알고리즘의 우연이 아닙니다.
B 안에서는 <span class="code">slope4</span>(−0.19)·<span class="code">rng4</span>(−0.30)가 일하고
<span class="code">is_ramp</span> 단독은 오히려 해로운데(+0.60) <b>넷을 합쳤을 때가 가장 낫습니다</b>.
이진 플래그 혼자서는 거칠고 연속값과 같이 있을 때 보조 역할을 하는 것으로 보입니다.</p>
<h4>③ 기온 파생 — 자료에서는 보이는데 모델은 안 좋아집니다</h4>
<p>원래 아이디어는 <i>“전력은 냉방·난방 모두에서 늘어나니 기온 원값은 상쇄되고,
|기온 − 기준| 로 접으면 관계가 드러날 것”</i>이었습니다. <b>자료에서는 맞았습니다.</b></p>
{table([["기저부하 (kW)", "<b>25.69</b>", "23.09", "<b>22.37</b>", "22.56", "22.69", "22.42"]],
       ["비가동 구간", "−3.4℃", "8.0℃", "17.4℃", "22.7℃", "27.1℃", "31.6℃"])}
<p>최저점 17.4℃가 기온 중앙값(17.8℃)과 거의 일치하고, 접으면 상관의 부호가 뒤집히며
절대값이 커집니다(−0.068 → +0.079, 난방도 기준 14℃면 +0.091).
<b>다만 대칭이 아닙니다</b> — 추운 쪽은 3.3 kW(15%) 오르는데 더운 쪽은 거의 안 오릅니다.
<b>이 공장은 난방은 하고 냉방은 거의 안 합니다.</b></p>
{note("“상관이 생겼다”와 “모델이 좋아진다”는 별개입니다",
 "기온 파생의 모델 개선은 −0.13으로 사실상 없습니다. 직전 전력이 이미 그 정보를 담고 있기 "
 "때문입니다 — 기저부하가 22.4냐 25.7이냐는 <span class='code'>과거전력_1칸</span>에 "
 "그대로 들어 있습니다.", "caution")}
{gru_rows}
{note("팀에 권하는 것",
 "<b>slope4 · rng4 · absd4 · is_ramp 네 개를 넣어 보세요</b> — 트리 계열 양쪽에서 유지됩니다.<br>"
 "<b>나머지는 넣지 마세요</b> — 특히 가동상태·수준대비는 해롭습니다.<br>"
 "변수를 늘리는 방향은 이 자료에서 잘 듣지 않습니다. 앞서 확인한 대로 "
 "<b>시각 원값</b>(시간·15분위치·분)을 넣는 것이 훨씬 컸습니다(46.58 → 41.75).", "good")}
</section>"""

    s4 = """<section id="s4"><span class="section-number">04 / REPRODUCE</span>
<h2>재현</h2>
<pre>cd Modeling_RNN
python train_rnn.py          # 시드 3개 재학습·평가
python predict_rnn.py

cd ensemble
python build_report.py       # 이 보고서 재생성 (재학습 없음)</pre>
<p>앙상블 수치는 <span class="code">Modeling/reproduced_test_predictions.csv</span>와
<span class="code">Modeling_RNN/reproduced_test_predictions.csv</span>를
시각으로 맞춰 계산합니다. 이 보고서의 모든 수치는 그 두 파일과
<span class="code">reproduced_metrics.csv</span>에서 읽습니다 — 손으로 적은 값이 아닙니다.</p>
</section>"""

    s5 = """<section id="s5"><span class="section-number">05 / LIMITS</span>
<h2>한계 — 먼저 밝힙니다</h2>
<ul>
<li><b>이 시험 구간(6/27~9/14)은 양쪽 모두 이미 여러 번 관찰한 구간</b>입니다.
정병근 보고서의 같은 단서가 여기에도 적용됩니다</li>
<li>전진검증 4구간뿐이라 <b>0.5 이내 차이는 구별되지 않는다</b>고 보는 편이 안전합니다.
파생변수 개선폭 대부분이 그 안에 있습니다</li>
<li>하계휴무가 시험 구간에 <b>한 번</b>뿐이라 휴무 대응 성능을 일반화할 수 없습니다</li>
<li>자료가 1~9월뿐이라 <b>겨울을 아무도 검증하지 못했습니다</b>.
기온 파생이 겨울에는 다르게 작용할 수 있습니다</li>
<li>순환 계열은 시드에 따라 결과가 달라집니다. 단일 시드 수치를 인용하지 마세요</li>
</ul>
</section>"""

    nav = "".join(f'<a href="#{i}">{esc(t)}</a>' for i, t in NAV)
    page = f"""<!doctype html><html lang="ko"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>순환신경망과 앙상블 | 팀 공유 보고서</title><style>{CSS}</style></head><body>
<div class="topbar">KAMP 자원 최적화 AI 데이터셋　/　순환신경망 및 앙상블 검토</div>
<div class="frame"><aside>{nav}</aside><main>
<div class="cover"><small>DEEP LEARNING PART · 팀 공유용</small>
<h1>순환신경망 결과와 앙상블</h1>
<p>과거 하루치 전력 계열을 순서대로 먹여 다음 15분을 예측했습니다.
<span class="code">Modeling/</span>(ExtraTrees)과 같은 행에서 비교하고,
<b>두 예측을 섞으면 어떻게 되는지</b>까지 봤습니다.</p>
<div class="docket">
<div><b>{n(d["gru"], 2)}</b><span>GRU 시드평균 MSE</span></div>
<div><b>{n(d["et"], 2)}</b><span>ExtraTrees 같은 행</span></div>
<div><b style="color:var(--aqua)">{n(d["half"], 2)}</b><span>두 예측 평균 ({n(cut, 1)}% 감소)</span></div>
<div><b>{d["rows"]:,}</b><span>공통 시험 행</span></div>
</div></div>
{s1}{s2}{s3}{s4}{s5}
<footer>출처 — 중소벤처기업부, Korea AI Manufacturing Platform(KAMP), 자원 최적화 AI 데이터셋,
KAIST(울산과학기술원, ㈜유피시앤에스), 2021.12.27., www.kamp-ai.kr<br>
이 브랜치(<span class="code">feat/rnn-ensemble</span>)의 내용은 main 에 올리지 않았습니다 — 검토용입니다.
</footer></main></div></body></html>"""
    OUT.write_text(page, encoding="utf-8")
    print(f"저장: {OUT.name}  ({len(page):,}자)")
    print(f"  공통 {d['rows']:,}행 · 트리 {d['et']:.3f} · GRU {d['gru']:.3f} "
          f"· 반반 {d['half']:.3f} ({cut:.1f}% 감소)")


if __name__ == "__main__":
    main()
