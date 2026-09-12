(() => {
  let D = window.FOUNDRY_DATA;
  const $ = (selector) => document.querySelector(selector);
  const features = Object.keys(D.specs);
  const label = (feature) => D.labels[feature] || feature;
  const fmt = (value, digits = 2) => Number(value ?? 0).toLocaleString("ko-KR", { maximumFractionDigits: digits });
  const pct = (value) => `${(Number(value || 0) * 100).toFixed(1)}%`;
  const range = (numbers) => [Math.min(...numbers), Math.max(...numbers)];
  const escape = (text) => String(text).replace(/[&<>\"]/g, (char) => ({ "&":"&amp;", "<":"&lt;", ">":"&gt;", '"':"&quot;" })[char]);
  const current = (rows, feature) => [...rows].reverse().find((row) => row.v[feature] != null)?.v[feature];
  const outside = (value, spec) => value != null && (value < spec.low || value > spec.high);
  const priority = (event) => {
    const breached = features.filter((feature) => outside(event.v[feature], D.specs[feature]));
    return { breached, critical: event.p >= D.threshold && breached.some((feature) => D.specs[feature].grade === "A"), warning: event.p >= D.threshold || breached.length > 0 };
  };
  const dates = D.events.map((event) => event.d);

  function filtered() {
    const from = $("#from-date").value;
    const to = $("#to-date").value;
    const risk = $("#risk-filter").value;
    return D.events.filter((event) => (!from || event.d >= from) && (!to || event.d <= to) && (risk === "all" || (risk === "high" && event.p >= D.threshold) || (risk === "defect" && event.y === 1)));
  }

  function svgLine(rows, valueOf, options = {}) {
    const host = $(options.host);
    const width = Math.max(host.clientWidth || 620, 320), height = Math.max(host.clientHeight || 250, 200);
    const pad = { l: 42, r: 12, t: 14, b: 28 };
    const values = rows.map(valueOf).filter(Number.isFinite);
    if (!values.length) return `<p class="muted">표시할 데이터가 없습니다.</p>`;
    let [min, max] = range(values); if (options.domain) [min, max] = options.domain; if (min === max) { min -= 1; max += 1; }
    const x = (index) => pad.l + index / Math.max(rows.length - 1, 1) * (width - pad.l - pad.r);
    const y = (value) => height - pad.b - (value - min) / (max - min) * (height - pad.t - pad.b);
    const points = rows.map((row, index) => `${x(index).toFixed(1)},${y(valueOf(row)).toFixed(1)}`).join(" ");
    const grid = [0, .25, .5, .75, 1].map((portion) => { const value = min + (max - min) * portion, yy = y(value); return `<line class="grid-line" x1="${pad.l}" x2="${width-pad.r}" y1="${yy}" y2="${yy}"/><text class="axis-label" x="2" y="${yy+4}">${options.percent ? (value*100).toFixed(0)+"%" : fmt(value)}</text>`; }).join("");
    const band = options.band ? `<rect class="band" x="${pad.l}" y="${y(options.band[1])}" width="${width-pad.l-pad.r}" height="${y(options.band[0])-y(options.band[1])}"/>` : "";
    const threshold = options.threshold == null ? "" : `<line class="threshold-line" x1="${pad.l}" x2="${width-pad.r}" y1="${y(options.threshold)}" y2="${y(options.threshold)}"/>`;
    const faults = options.faults ? rows.map((row, index) => row.y ? `<circle class="fault" cx="${x(index)}" cy="${y(valueOf(row))}" r="3.2"/>` : "").join("") : "";
    const labels = `<text class="axis-label" x="${pad.l}" y="${height-6}">${rows[0]?.d || ""}</text><text class="axis-label" text-anchor="end" x="${width-pad.r}" y="${height-6}">${rows.at(-1)?.d || ""}</text>`;
    return `<svg class="svg-chart" viewBox="0 0 ${width} ${height}" preserveAspectRatio="none" aria-hidden="true">${grid}${band}${threshold}<polyline class="${options.kind === "risk" ? "risk-path" : "process-path"}" points="${points}"/>${faults}${labels}</svg>`;
  }

  function renderOverview(rows) {
    const last = rows.at(-1), riskEvents = rows.filter((event) => event.p >= D.threshold), defects = rows.filter((event) => event.y), alerts = rows.filter((event) => priority(event).warning);
    const meanRisk = rows.reduce((sum, event) => sum + event.p, 0) / Math.max(rows.length, 1);
    $("#row-summary").textContent = `표시 데이터 ${fmt(rows.length)}건 · 모델 임계값 ${D.threshold}`;
    $("#command-strip").innerHTML = [
      ["현재 이벤트", last ? pct(last.p) : "—", last?.t || ""], ["실측 불량률", pct(defects.length / Math.max(rows.length, 1)), `${fmt(defects.length)}건`], ["고위험 판정", pct(riskEvents.length / Math.max(rows.length, 1)), `${fmt(riskEvents.length)}건`], ["평균 위험", pct(meanRisk), "예측 확률 평균"], ["예방 경보", fmt(alerts.length), "예측·스펙 복합"]
    ].map((item, index) => `<div class="command-item ${index === 0 ? "emphasis" : ""}"><span>${item[0]}</span><b>${item[1]}</b><span>${item[2]}</span></div>`).join("");
    $("#risk-chart").innerHTML = svgLine(rows, (event) => event.p, { host: "#risk-chart", kind: "risk", percent: true, threshold: D.threshold, faults: true, domain: [0, 1] });
    const status = last ? priority(last) : { breached: [], critical: false, warning: false };
    const state = status.critical ? ["즉시 확인", "복합 위험", "주조 조건 변경 전, 아래 항목과 센서 상태를 확인하세요."] : status.warning ? ["선제 점검", "주의 신호", "불량 예측 또는 후보 운영구간 이탈을 확인했습니다."] : ["안정 관찰", "현재 정상", "현재 이벤트는 위험 임계값과 후보 운영구간에서 안정적입니다."];
    $("#action-panel").className = `action-panel ${status.warning ? "caution" : "safe"}`;
    $("#action-panel").innerHTML = `<p class="eyebrow">Recommended action</p><h3>${state[0]}<br>${state[1]}</h3><p>${state[2]}</p><div class="next-step"><b>다음 조치</b><br>${status.breached.slice(0, 3).map((feature) => `${label(feature)}: 실제값·센서·SOP 확인`).join("<br>") || "최근 위험도 추이를 유지 관찰"}</div>`;
    $("#feature-signals").innerHTML = features.slice(0, 6).map((feature) => { const spec = D.specs[feature], value = current(rows, feature), alert = outside(value, spec), position = Math.max(0, Math.min(100, (value - spec.low) / Math.max(spec.high - spec.low, .001) * 100)); return `<div class="signal-row"><span>${escape(label(feature))} <b class="badge ${spec.grade.toLowerCase()}">${spec.grade}</b></span><span class="signal-track"><b></b><i style="left:${position}%"></i></span><span class="signal-state ${alert ? "alert" : ""}">${alert ? "후보 이탈" : "후보 내"}</span></div>`; }).join("");
    const recent = [...rows].reverse().filter((event) => event.p >= D.threshold).slice(0, 6);
    $("#recent-events").innerHTML = recent.map((event) => { const p = priority(event); return `<tr><td>${event.t.slice(5)}</td><td>${pct(event.p)}</td><td><span class="badge ${event.y ? "a" : "normal"}">${event.y ? "불량" : "미확정"}</span></td><td>${p.breached.slice(0,2).map(label).join(", ") || "예측 위험"}</td></tr>`; }).join("") || `<tr><td colspan="4" class="muted">해당 기간에 고위험 이벤트가 없습니다.</td></tr>`;
  }

  function renderAnalysis(rows) {
    if (!rows.length) {
      ["#process-chart", "#distribution-chart", "#scatter-chart"].forEach((id) => { $(id).innerHTML = `<p class="muted">필터 조건에 맞는 데이터가 없습니다.</p>`; });
      return;
    }
    const feature = $("#feature-select").value || features[0], spec = D.specs[feature], values = rows.map((event) => event.v[feature]).filter(Number.isFinite);
    $("#series-title").textContent = `${label(feature)} 시계열`;
    $("#process-chart").innerHTML = svgLine(rows, (event) => event.v[feature], { host: "#process-chart", band: [spec.low, spec.high], domain: [Math.min(...values, spec.low), Math.max(...values, spec.high)] });
    const good = rows.filter((event) => !event.y).map((event) => event.v[feature]).filter(Number.isFinite), bad = rows.filter((event) => event.y).map((event) => event.v[feature]).filter(Number.isFinite);
    $("#distribution-chart").innerHTML = distributionSvg(good, bad, "#distribution-chart", label(feature));
    $("#scatter-chart").innerHTML = scatterSvg(rows, "#scatter-chart");
  }

  function distributionSvg(good, bad, hostId, title) {
    const host = $(hostId), width = Math.max(host.clientWidth || 500, 280), height = Math.max(host.clientHeight || 250, 200), pad = { l: 42, r: 16, t: 18, b: 30 }, vals = good.concat(bad); if (!vals.length) return "";
    let [min, max] = range(vals); if (min === max) max += 1; const y = (v) => height-pad.b-(v-min)/(max-min)*(height-pad.t-pad.b), q = (arr,n) => { const s=[...arr].sort((a,b)=>a-b); return s[Math.floor((s.length-1)*n)] ?? min; };
    const box = (arr, x, color, name) => { const [a,b,c]=[q(arr,.25),q(arr,.5),q(arr,.75)]; return `<line stroke="${color}" x1="${x}" x2="${x}" y1="${y(q(arr,.05))}" y2="${y(q(arr,.95))}"/><rect x="${x-26}" y="${y(c)}" width="52" height="${y(a)-y(c)}" fill="${color}55" stroke="${color}"/><line stroke="${color}" stroke-width="2" x1="${x-26}" x2="${x+26}" y1="${y(b)}" y2="${y(b)}"/><text class="axis-label" text-anchor="middle" x="${x}" y="${height-6}">${name}</text>`; };
    return `<svg class="svg-chart" viewBox="0 0 ${width} ${height}" preserveAspectRatio="none"><text class="axis-label" x="${pad.l}" y="12">${escape(title)}</text>${box(good,width*.35,"#39795d","양품")}${box(bad,width*.68,"#c7463a","불량")}</svg>`;
  }

  function scatterSvg(rows, hostId) {
    const host = $(hostId), width = Math.max(host.clientWidth || 500, 280), height = Math.max(host.clientHeight || 250, 200), pad = { l: 42, r: 15, t: 15, b: 32 }, xKey = "injection_pressure", yKey = "bottom_temp2", subset = rows.filter((_, index) => index % Math.ceil(rows.length / 1200) === 0), xs=subset.map(e=>e.v[xKey]).filter(Number.isFinite),ys=subset.map(e=>e.v[yKey]).filter(Number.isFinite); if(!xs.length||!ys.length)return ""; const [x0,x1]=range(xs),[y0,y1]=range(ys), x=v=>pad.l+(v-x0)/(x1-x0||1)*(width-pad.l-pad.r),y=v=>height-pad.b-(v-y0)/(y1-y0||1)*(height-pad.t-pad.b); return `<svg class="svg-chart" viewBox="0 0 ${width} ${height}" preserveAspectRatio="none">${subset.map(e=>Number.isFinite(e.v[xKey])&&Number.isFinite(e.v[yKey])?`<circle class="${e.y?"scatter-bad":"scatter-good"}" cx="${x(e.v[xKey])}" cy="${y(e.v[yKey])}" r="2.2"/>`:"").join("")}<text class="axis-label" x="${pad.l}" y="${height-5}">주입 압력</text><text class="axis-label" text-anchor="end" x="${width-pad.r}" y="${height-5}">하부 온도 2</text></svg>`;
  }

  function renderSpecs(rows) {
    $("#spec-rows").innerHTML = features.map((feature) => { const spec = D.specs[feature], value = current(rows, feature), alert = outside(value, spec), position = Math.max(0, Math.min(100, (value-spec.low)/Math.max(spec.high-spec.low,.001)*100)); return `<tr><td><span class="badge ${spec.grade.toLowerCase()}">${spec.grade}</span></td><td>${escape(label(feature))}</td><td>${pct(spec.importance)}</td><td><span class="spec-range"><small>${fmt(spec.low)}</small><span class="range-track"><b></b><i style="left:${position}%"></i></span><small>${fmt(spec.high)}</small></span></td><td>${fmt(value)}</td><td><span class="badge ${alert ? "warning" : "normal"}">${alert ? "후보 이탈" : "후보 내"}</span></td><td><span class="stage">Shadow 후보</span></td></tr>`; }).join("");
  }

  function renderAlerts(rows) {
    const alerts = [...rows].reverse().filter((event) => priority(event).warning).slice(0, 40);
    $("#alert-rows").innerHTML = alerts.map((event, index) => { const p = priority(event), key = `foundry-action-${event.t}`, done = localStorage.getItem(key); return `<tr><td><span class="badge ${p.critical ? "critical" : "warning"}">${p.critical ? "Critical" : "Warning"}</span></td><td>${event.t}</td><td>${p.critical ? "복합 예방 경보" : "고위험 또는 후보 이탈"}</td><td>예측 ${pct(event.p)}${p.breached.length ? ` · ${p.breached.map(label).join(", ")}` : ""}</td><td><button class="action-button" data-alert-key="${key}" ${done ? "disabled" : ""}>${done ? "확인됨" : "확인 및 조치 기록"}</button></td></tr>`; }).join("") || `<tr><td colspan="5" class="muted">해당 기간에 예방 경보가 없습니다.</td></tr>`;
    document.querySelectorAll("[data-alert-key]").forEach((button) => button.addEventListener("click", () => { localStorage.setItem(button.dataset.alertKey, new Date().toISOString()); button.textContent = "확인됨"; button.disabled = true; }));
  }

  function render() { const rows = filtered(); renderOverview(rows); renderAnalysis(rows); renderSpecs(rows); renderAlerts(rows); }
  function switchView(id) { document.querySelectorAll(".view").forEach((view) => view.classList.toggle("active", view.id === id)); document.querySelectorAll("[data-view]").forEach((link) => link.classList.toggle("active", link.dataset.view === id)); requestAnimationFrame(render); }

  $("#from-date").value = dates[0]; $("#to-date").value = dates.at(-1); $("#data-period").textContent = `${D.source.from.slice(0,10)} ~ ${D.source.to.slice(0,10)}`;
  features.forEach((feature) => $("#feature-select").insertAdjacentHTML("beforeend", `<option value="${feature}">${escape(label(feature))}</option>`));
  ["#from-date", "#to-date", "#risk-filter", "#feature-select"].forEach((id) => $(id).addEventListener("change", render));
  $("#reset-filter").addEventListener("click", () => { $("#from-date").value = dates[0]; $("#to-date").value = dates.at(-1); $("#risk-filter").value = "all"; render(); });
  document.querySelectorAll("[data-view]").forEach((link) => link.addEventListener("click", (event) => { event.preventDefault(); switchView(link.dataset.view); history.replaceState(null, "", `#${link.dataset.view}`); }));
  window.addEventListener("resize", () => render());
  async function start() {
    try {
      const response = await fetch("/api/dashboard");
      if (response.ok) D = await response.json();
    } catch (_) {
      // ponytail: data.js remains a local-file fallback; use the API whenever FastAPI is running.
    }
    switchView(location.hash.slice(1) || "overview");
  }
  start();
})();
