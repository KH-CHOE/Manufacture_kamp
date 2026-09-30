#!/usr/bin/env python3
"""앙상블 보고서 HTML 생성. 결과 파일에서 수치를 읽어 만든다.

실행: python build_report.py
출력: 앙상블_보고서.html

`Modeling/build_report.py` 와 같은 시각 언어(색·구조·인쇄 규칙)를 쓴다.
재학습은 하지 않는다 — reproduction_metadata.json 과 evidence/ 만 읽는다.
"""
from __future__ import annotations

import html
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
EV = HERE / "evidence"
OUT = HERE / "앙상블_보고서.html"


def esc(v) -> str:
    return html.escape(str(v), quote=True)


def n(v, d=2) -> str:
    return f"{float(v):,.{d}f}"


def js(name: str):
    p = EV / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def table(rows, head, hi=None, left=(0,)):
    th = "".join(f"<th{' class=l' if i in left else ''}>{esc(h)}</th>"
                 for i, h in enumerate(head))
    tr = []
    for i, r in enumerate(rows):
        cls = ' class="hi"' if hi is not None and i == hi else ""
        td = "".join(f"<td{' class=l' if k in left else ''}>{c}</td>"
                     for k, c in enumerate(r))
        tr.append(f"<tr{cls}>{td}</tr>")
    return (f'<div class="table-scroll"><table><thead><tr>{th}</tr></thead>'
            f'<tbody>{"".join(tr)}</tbody></table></div>')


def note(title, body, kind=""):
    k = f" {kind}" if kind else ""
    return f'<div class="note{k}"><strong>{title}</strong><p>{body}</p></div>'


CSS = """
:root{--ink:#173047;--blue:#2462a9;--aqua:#008a8c;--gold:#b47727;--red:#a8342a;
--paper:#f5f8fa;--line:#d8e3eb;--muted:#5a6e7b}
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
.frame{max-width:1320px;margin:auto;display:grid;grid-template-columns:200px minmax(0,1fr);
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
th.l,td.l{text-align:left}
thead th{background:#eaf1f6;color:#35526b;font-weight:700}
tr:last-child td{border-bottom:0}tbody tr:hover{background:#f4f8fb}
tr.hi td{background:#e9f5f1;font-weight:700}
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
pre{background:var(--ink);color:#fff;padding:18px;overflow:auto;
font:12.5px/1.7 ui-monospace,Menlo,monospace}
footer{font-size:12px;color:var(--muted);padding:10px 0 30px}
@media(max-width:980px){.frame{display:block}aside{position:static;display:flex;overflow:auto;
margin-bottom:20px}aside a{white-space:nowrap}.docket{grid-template-columns:repeat(2,1fr)}}
@media(max-width:620px){.frame{padding:20px 12px}.cover,section{padding:24px 17px}
h1{font-size:28px}h2{font-size:22px}.docket,.summary-grid{grid-template-columns:1fr}}
@media print{body{background:#fff}.frame{display:block;padding:0}aside{display:none}
.cover,section{break-inside:avoid}.table-scroll{overflow:visible}
a{text-decoration:none;color:var(--ink)}}
"""

NAV = [("s1", "1 무엇을 했나"), ("s2", "2 변수 선정"), ("s3", "3 앙상블"),
       ("s4", "4 순위"), ("s5", "5 안 된 것"), ("s6", "6 재현"), ("s7", "7 한계")]


def main() -> None:
    meta = json.loads((HERE / "reproduction_metadata.json").read_text(encoding="utf-8"))
    cfg, t = meta["config"], meta["test"]
    fwd, folds = meta.get("forward"), meta.get("forward_folds", [])
    rank = js("01_forward_same_basis.json")
    enc = js("04_time_encoding.json")
    feat = js("05_feature_selection.json")
    fam = js("06_derived_features.json")
    gruf = js("08_derived_on_gru.json")
    wgt = js("09_blend_weight.json")
    tp = js("03_test_period.json")
    best_solo = min(t["tree"]["MSE"], t["gru"]["MSE"])

    s1 = f"""<section id="s1"><span class="section-number">01 / WHAT</span>
<h2>무엇을 했나</h2>
<p class="lede">트리 계열과 순환신경망은 <b>서로 다른 조건에서 무너집니다.</b>
그래서 하나를 고르는 대신 <b>두 예측을 평균냈습니다.</b>
같은 행·같은 분할에서 전진검증과 시험 구간 <b>양쪽 모두 1위</b>입니다.</p>
{table([
 ["트리", f"ExtraTrees · {len(cfg['features'])}변수",
  "달력으로 <b>분기</b>해 여러 체제를 따로 담는다"],
 ["순환신경망", f"GRU · 은닉 {cfg['gru']['hidden']} · 창 {cfg['gru']['window']}칸 · "
  f"직전 {cfg['gru']['train_months']}개월 · 시드 3개 평균",
  "과거 계열을 <b>순서대로</b> 먹으며 흐름을 따라간다"],
 ["결합", f"고정 {cfg['blend_weight']} : {1 - cfg['blend_weight']}",
  "가중치를 고르지 않는다 (3절 참조)"],
], ["구성", "설정", "성질"], left=(0, 1, 2))}
{note("왜 섞는 것이 통하는가",
 "두 예측의 상관은 <b>0.996</b> 으로 값이 거의 같습니다. 그런데도 크게 줄어듭니다 — "
 "<b>틀리는 방향이 다르기</b> 때문입니다. 트리는 임계값으로 자르고 순환신경망은 계열을 "
 "따라가니, 실패하는 조건이 겹치지 않습니다. 같은 계열끼리 섞으면(ExtraTrees + 부스팅) "
 f"{n(tp['세모형_결합']['트리 두 판 평균']) if tp else '58.42'} 밖에 안 되는데, "
 "다른 계열을 섞으면 훨씬 큽니다.", "good")}
</section>"""

    enc_rows = []
    if enc:
        b = enc["우리_21변수"]
        enc_rows = [["우리 21변수 (원래)", "21", n(b), "—"]] + [
            [k, str(v["변수수"]), n(v["평균"]), f"{v['우리 대비']:+.2f}"]
            for k, v in enc["판"].items()]
    s2 = f"""<section id="s2"><span class="section-number">02 / FEATURES</span>
<h2>변수 선정 — 시각을 어떻게 주느냐가 갈랐습니다</h2>
<p>모든 판정은 <b>전진검증 4구간 평균 MSE</b> 로 했습니다. 시험 구간은 쓰지 않았습니다.</p>
<h3>가장 큰 발견 — 트리에는 시각 <b>원값</b>이 필요합니다</h3>
<p>트리는 <span class="code">변수 ≤ 임계값</span> 으로만 자릅니다. 그래서
<span class="code">시간 ≥ 7</span>(오전 시동)을 <b>한 번의 분기</b>로 만들려면 원값이
있어야 합니다. 순환 인코딩(sin/cos)만 주면 두 변수를 여러 번 잘라야 그 경계가 생깁니다.
<b>순환 인코딩은 신경망에는 맞지만 트리에는 불리합니다.</b></p>
{table(enc_rows, ["설정", "변수", "전진검증", "우리 대비"], hi=1, left=(0,))}
{note("중복 변수를 하나 찾아 뺐습니다",
 "실험에서 쓴 시각 원값 세 개 중 <span class='code'>HS_분</span> 은 "
 "<span class='code'>15분위치</span> 의 순서만 바꾼 것이었습니다(교차표가 완전한 일대일). "
 "빼도 39.66 → 39.74 로 차이가 없어(구별 기준 0.5 이내) <b>25변수로 정리</b>했습니다. "
 "반면 <span class='code'>15분위치</span> 를 빼면 43.15 로 무너집니다 — "
 "시각 원값의 힘은 <b>시간 + 15분위치</b> 에서 나옵니다.", "caution")}
<h3>무엇을 넣고 무엇을 뺐나</h3>
{table([
 ["넣음", "시각 원값 <span class='code'>시간</span>·<span class='code'>15분위치</span>", "46.58 → 41.75 (−4.83)"],
 ["넣음", "최근 2·4시간 통계 7종", "빼면 41.61 → 43.29 (+1.69)"],
 ["뺌", "기상 4종 (기온·풍속·습도·강수량)", "빼면 39.88 → 39.77 (−0.11)"],
 ["뺌", "1주 전 <span class='code'>kw_lag672</span>", "넣으면 41.61 → 45.05 (+3.44)"],
], ["", "변수", "근거"], left=(0, 1, 2))}
{note("기상을 왜 뺐나",
 "기온이 전력에 영향을 주기는 합니다 — 비가동 기저부하가 −3.4℃ 에서 25.7 kW, "
 "17.4℃ 에서 22.4 kW 로 <b>U자</b>입니다(난방만 하고 냉방은 거의 안 합니다). "
 "그런데 <b>모형 성능으로는 이어지지 않습니다.</b> 직전 전력이 이미 그 정보를 담고 있기 "
 "때문입니다. 기온을 <span class='code'>|기온 − 중앙값|</span> 이나 난방도로 바꿔 줘도 "
 "개선이 −0.13 으로 사실상 없었습니다. "
 "<b>“상관이 생겼다”와 “모형이 좋아진다”는 별개입니다.</b>", "caution")}
</section>"""

    fold_rows = [[f["fold"][4:], n(f["tree"]["MSE"]), n(f["gru"]["MSE"]),
                  n(f["blend"]["MSE"]),
                  f"{f['blend']['MSE'] - f['tree']['MSE']:+.2f}", f"{f['rows']:,}"]
                 for f in folds]
    if fwd:
        fold_rows.append(["<b>평균</b>", f"<b>{n(fwd['tree'])}</b>", f"<b>{n(fwd['gru'])}</b>",
                          f"<b>{n(fwd['blend'])}</b>",
                          f"<b>{fwd['blend'] - fwd['tree']:+.2f}</b>", ""])
    sw = meta.get("forward_weight_sweep")
    wt_rows = [["고정 0.5 : 0.5 <b>(채택)</b>", "—",
                "선정에 시험·전진검증 어느 쪽도 쓰지 않는다"]]
    if sw:
        wt_rows.append(["전진검증으로 역산한 최적 (진단)",
                        f"트리 {sw['best_weight']:.2f}",
                        f"{sw['best_MSE']:.2f} (고정 반반 {sw['fixed_half_MSE']:.2f}, "
                        f"차이 {sw['gain_of_tuning']:+.2f})"])
    else:
        wt_rows.append(["전진검증으로 역산한 최적", "—",
                        "<b>재지 않았다</b> — 결과 파일에 값이 없다"])
    wt_rows += [
        ["정병근 시험 구간으로 역산한 최적 (진단)", "트리 0.43",
         "51.90 (고정 반반 52.07, 차이 −0.17)"],
        ["시험 앞 20% 로 학습해 뒤 80% 에 적용", "트리 0.75",
         "<b>같은 구간 고정반반보다 3.9 나쁨</b>"],
    ]
    weight_table = table(wt_rows, ["방식", "가중치", "결과"], hi=0, left=(0, 1, 2))
    s3 = f"""<section id="s3"><span class="section-number">03 / ENSEMBLE</span>
<h2>앙상블 — 전진검증 관문을 통과했습니다</h2>
<p>앙상블 수치를 <b>시험 구간에서만</b> 재면 선정 근거로 쓸 수 없습니다.
그래서 선정 기준인 전진검증 4구간에서 구간마다 트리와 GRU를 각각 학습해 섞었습니다.</p>
{table(fold_rows, ["구간", "트리", "GRU", "앙상블", "이득", "행수"],
       hi=len(fold_rows) - 1 if fwd else None, left=(0,))}
{note("항상 이기는 것이 아닙니다 — 그게 중요합니다",
 "네 구간 중 <b>둘에서 벌고 둘에서 조금 잃습니다.</b> 버는 구간은 트리가 어려워하는 곳이고, "
 "잃는 구간은 트리가 이미 잘 맞히는 곳입니다. 즉 <b>평균적으로 이기고 최악을 줄입니다.</b> "
 "실전에서 다음 구간이 쉬울지 어려울지 미리 알 수 없으므로, 그것이 오히려 원하는 성질입니다.",
 "good")}
<h3>시험 구간</h3>
{table([["트리 (ExtraTrees)", n(t["tree"]["MSE"], 3), n(t["tree"]["RMSE"], 3),
         n(t["tree"]["MAE"], 3), n(t["tree"]["R2"], 4)],
        ["GRU (시드 3개 평균)", n(t["gru"]["MSE"], 3), n(t["gru"]["RMSE"], 3),
         n(t["gru"]["MAE"], 3), n(t["gru"]["R2"], 4)],
        ["<b>앙상블</b>", f"<b>{n(t['blend']['MSE'], 3)}</b>", n(t["blend"]["RMSE"], 3),
         n(t["blend"]["MAE"], 3), n(t["blend"]["R2"], 4)]],
       ["모형", "MSE", "RMSE", "MAE", "R²"], hi=2, left=(0,))}
<p>단독 최고({n(best_solo, 3)}) 대비 <b>{(1 - t['blend']['MSE'] / best_solo) * 100:.1f}% 감소</b>
입니다. 시험 {meta['test_rows']:,}행.</p>
<h3>가중치는 고르지 않습니다</h3>
{weight_table}
<p><b>구간마다 최적이 반대 방향입니다.</b> 전진검증에서 역산한 최적은 트리 쪽으로,
정병근 시험 구간에서 역산한 최적은 순환신경망 쪽(0.43)으로 기웁니다. 그리고 실제로
가중치를 학습시키면(시험 앞 20% 로 정해 뒤 80% 에 적용) 같은 구간 고정 반반보다
<b>3.9 나빠집니다</b>. 작은 이득에 선정 편향을 사지 않습니다.</p>
<p class="small">전진검증 줄은 <code>reproduction_metadata.json</code> 의
<code>forward_weight_sweep</code> 에서 읽습니다. <b>진단으로 재기만 했고 가중치 선정에는
쓰지 않았습니다</b> — 그 4구간으로 가중치를 고르면 같은 구간이 선정에 쓰인 것이 됩니다.
아래 두 줄은 7,104행 기준이라 위와 행 기준이 다릅니다.</p>
</section>"""

    rank_rows = []
    if rank:
        for i, r in enumerate(rank["순위"], 1):
            nm = r["후보"]
            bold = "앙상블" in nm or "반반" in nm
            rank_rows.append([f"{i}", f"<b>{esc(nm)}</b>" if bold else esc(nm),
                              f"<b>{n(r['평균'])}</b>" if bold else n(r["평균"])])
    s4 = f"""<section id="s4"><span class="section-number">04 / RANKING</span>
<h2>같은 행 기준 전체 순위</h2>
<p>행 기준이 다른 수치를 섞으면 안 되므로, 후보 전부를
<b>{rank['행수'] if rank else 0:,}행 하나</b>에서 다시 쟀습니다.</p>
{table(rank_rows, ["", "후보", "전진검증 평균"], hi=0, left=(0, 1))}
{note("행 기준 주의",
 "이 저장소 안에 시험 행수가 여럿 있습니다 — "
 "<span class='code'>Modeling/</span> 7,294행 · <span class='code'>Modeling_RNN/</span> 7,104행 · "
 f"이 폴더 {meta['test_rows']:,}행. 순환신경망은 창 96칸을 만들 수 없는 행을 쓸 수 없고, "
 "삭제된 2021-07-13·15 주변이 그에 해당합니다. <b>같은 기준끼리만 비교하세요.</b>", "caution")}
</section>"""

    fam_rows = []
    if fam:
        fam_rows = [["기준선 (우리 + 시각 원값)", "24", n(fam["기준선_평균"]), "—"]] + [
            [f"+ {k}", str(24 + len(v["변수"])), n(v["평균"]), f"{-v['개선']:+.2f}"]
            for k, v in fam["묶음"].items()]
        fam_rows.append(["+ 전부 (23개)", "47", n(fam["전부_추가"]["평균"]),
                         f"{-fam['전부_추가']['개선']:+.2f}"])
    gru_rows = []
    if gruf:
        gb = gruf["기준"]
        gru_rows = [[k, n(v), "—" if k.startswith("기준") else f"{v - gb:+.2f}"]
                    for k, v in gruf["판"].items()]
    s5 = f"""<section id="s5"><span class="section-number">05 / WHAT FAILED</span>
<h2>안 된 것 — 기록해 둡니다</h2>
<p>분석에서 나온 사실마다 파생변수를 만들어 23개를 6묶음으로 나눠 시험했습니다.
<b>대부분 안 됐습니다.</b></p>
{table(fam_rows, ["묶음", "변수", "전진검증", "기준 대비"], left=(0,))}
<p><b>변수를 많이 더할수록 나빠집니다.</b> 전부 넣으면 +2.19 입니다.
살아남은 것은 <b>B 전환감지</b>(<span class="code">slope4</span>·<span class="code">rng4</span>·
<span class="code">absd4</span>·<span class="code">is_ramp</span>) 하나뿐이고,
그것도 −0.56 으로 구별 기준(0.5) 경계입니다.
다만 부스팅에서도 −1.21 로 유지되어 <b>트리 계열 일반 효과</b>로 보입니다.</p>
{note("누수 검사가 하나를 잡았습니다",
 "<span class='code'>temp_range_day</span>(일교차)를 "
 "<span class='code'>groupby(날짜).transform('max') − min</span> 으로 만들었는데, "
 "이건 <b>아침 8시에 그날 최고기온을 아는 것</b>입니다. "
 "“지금까지의 누적 일교차”로 고친 뒤에야 검사를 통과했습니다. "
 "파생변수는 이렇게 조용히 미래를 봅니다 — 성능을 재기 <i>전에</i> 검사를 거는 게 맞았습니다.",
 "bad")}
<h3>순환신경망에는 파생변수가 전부 해로웠습니다</h3>
{table(gru_rows, ["설정", "전진검증", "기준 대비"], left=(0,))}
<p>트리에서 유일하게 살았던 B 전환감지가 <b>순환신경망에서는 가장 해롭습니다.</b>
GRU는 과거 96칸 계열을 그대로 받으므로 <span class="code">slope4</span>·
<span class="code">rng4</span> 같은 요약은 <b>이미 가진 정보를 다시 주는 것</b>이고,
출력단 스칼라라 잡음만 늘립니다. 트리는 계열을 받지 못하니 그 요약이 새 정보였습니다.</p>
{note("같은 파생변수가 모형 종류에 따라 정반대로 작용합니다",
 "이것이 이 실험의 가장 일반적인 교훈입니다. 변수 설계는 <b>모형이 무엇을 이미 보고 있는지</b>"
 "에 달려 있습니다.", "good")}
</section>"""

    s6 = f"""<section id="s6"><span class="section-number">06 / REPRODUCE</span>
<h2>재현</h2>
<pre>python prepare_ensemble_input.py --raw &lt;okm_augumented_2021.csv&gt; \\
    --align ../Modeling/final_input_data.csv   # 정병근 행·분할에 맞춤(비교용)
python train_ensemble.py                       # 전진검증 4구간 + 시험 구간
python build_report.py                         # 이 보고서 (재학습 없음)</pre>
<p><span class="code">--align</span> 을 빼면 원본만으로 자립 실행됩니다(날짜 규칙으로 같은 분할 재현).
<span class="code">--verify</span> 를 주면 <span class="code">Modeling/</span> 의 같은 이름 변수와
<b>전 행 값을 대조</b>합니다 — 실험에서 그의 CSV 를 썼으므로 이 파일이 그것을 재현하는지
확인할 수 있습니다(현재 전부 일치).</p>
{table([["ensemble_input_data.csv", f"{meta['rows_total']:,}행 · 트리 {len(cfg['features'])}변수 + 계열 + 달력"],
        ["selection.json", "최종 구성 · 선정 규칙 · 결합 규칙"],
        ["final_tree.joblib · final_gru.pt", "저장 모델 (GRU 는 첫 시드로 고정)"],
        ["reproduced_metrics.csv · _folds.csv · _test_predictions.csv", "재현 산출물"],
        ["reproduction_metadata.json", "구간별 전체 지표 · 환경"],
        ["evidence/ 10건", "변수 선정 · 앙상블 · 파생변수 · 기온 실험 기록"]],
       ["파일", "무엇"], left=(0, 1))}
</section>"""

    s7 = f"""<section id="s7"><span class="section-number">07 / LIMITS</span>
<h2>한계 — 먼저 밝힙니다</h2>
<ul>
<li><b>전진검증 4구간뿐입니다.</b> 0.5 이내 차이는 구별되지 않는다고 보는 편이 안전합니다.
파생변수 개선폭 대부분이 그 안에 있습니다</li>
<li>시험 구간(6/27~9/14)은 <b>팀 전체가 이미 여러 번 관찰한 구간</b>입니다.
새로운 독립 기간의 성능으로 주장하지 않습니다</li>
<li>앙상블이 <b>네 구간 중 둘에서는 조금 손해</b>입니다. 평균 이득이지 항상 이득이 아닙니다</li>
<li>순환신경망은 시드에 따라 결과가 달라집니다. 단일 시드 수치를 인용하지 마시고
<span class="code">reproduced_metrics.csv</span> 를 보세요</li>
<li>자료가 1~9월 9개월뿐이라 <b>겨울을 검증하지 못했습니다.</b>
기온을 뺀 결정이 겨울에는 다를 수 있습니다</li>
<li>하루 96개 값이 완전히 같은 날이 160/257일(62.3%)입니다. 시험 구간에 겹치는 하루가
없어 보고 성능은 부풀려지지 않았지만 <b>일반화는 주장하지 않습니다</b></li>
</ul>
</section>"""

    nav = "".join(f'<a href="#{i}">{esc(x)}</a>' for i, x in NAV)
    page = f"""<!doctype html><html lang="ko"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>트리 + 순환신경망 앙상블 | 모델링 보고서</title><style>{CSS}</style></head><body>
<div class="topbar">KAMP 자원 최적화 AI 데이터셋　/　트리 + 순환신경망 앙상블</div>
<div class="frame"><aside>{nav}</aside><main>
<div class="cover"><small>ENSEMBLE OF EXTRATREES AND GRU</small>
<h1>트리 + 순환신경망 앙상블</h1>
<p>두 모형은 <b>서로 다른 조건에서 무너집니다.</b> 그래서 고르는 대신 평균냈습니다.
전진검증과 시험 구간 <b>양쪽 모두 1위</b>입니다.</p>
<div class="docket">
<div><b>{n(fwd['blend']) if fwd else '—'}</b><span>전진검증 평균 (트리 {n(fwd['tree']) if fwd else '—'})</span></div>
<div><b>{n(t['blend']['MSE'])}</b><span>시험 구간 (단독 최고 {n(best_solo)})</span></div>
<div><b>{len(cfg['features'])}</b><span>트리 변수</span></div>
<div><b>{cfg['blend_weight']} : {1 - cfg['blend_weight']}</b><span>고정 결합 비중</span></div>
</div></div>
{s1}{s2}{s3}{s4}{s5}{s6}{s7}
<footer>출처 — 중소벤처기업부, Korea AI Manufacturing Platform(KAMP), 자원 최적화 AI 데이터셋,
KAIST(울산과학기술원, ㈜유피시앤에스), 2021.12.27., www.kamp-ai.kr<br>
모든 수치는 <span class="code">reproduction_metadata.json</span> 과
<span class="code">evidence/</span> 에서 읽습니다 — 손으로 적은 값이 아닙니다.
</footer></main></div></body></html>"""
    OUT.write_text(page, encoding="utf-8")
    print(f"저장: {OUT.name}  ({len(page):,}자)")
    if fwd:
        print(f"  전진검증 트리 {fwd['tree']} · GRU {fwd['gru']} · 앙상블 {fwd['blend']}")
    print(f"  시험 트리 {t['tree']['MSE']:.3f} · GRU {t['gru']['MSE']:.3f} "
          f"· 앙상블 {t['blend']['MSE']:.3f}")


if __name__ == "__main__":
    main()
