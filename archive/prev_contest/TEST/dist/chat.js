/* AI 어시스턴트 드로어 — POST /api/chat 호출. 서버측 도구가 현황·원인·비용·예측·조치를 응답. */
(() => {
  const $ = (s) => document.querySelector(s);
  const esc = (t) => String(t).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
  const history = [];
  let busy = false;

  const SAMPLES = [
    "지금 공정 상태 요약해줘",
    "가장 위험한 이벤트 원인이 뭐야?",
    "임계값을 0.5로 낮추면 비용이 어떻게 돼?",
    "주입압력 700, 금형온도 620이면 불량 위험은?",
  ];

  const body = () => $("#chat-body");

  function addMsg(role, text, pending = false) {
    const el = document.createElement("div");
    el.className = `chat-msg ${role}${pending ? " pending" : ""}`;
    el.textContent = text;
    body().appendChild(el);
    body().scrollTop = body().scrollHeight;
    return el;
  }

  function addTools(trace) {
    if (!trace || !trace.length) return;
    const d = document.createElement("details");
    d.className = "chat-tools";
    const names = trace.map((t) => t.name).join(", ");
    d.innerHTML = `<summary>🔧 근거 도구 ${trace.length}건 · ${esc(names)}</summary>` +
      trace.map((t) => `<pre><span class="chat-tool-name">${esc(t.name)}(${esc(JSON.stringify(t.args))})</span>\n${esc(JSON.stringify(t.result, null, 2))}</pre>`).join("");
    body().appendChild(d);
    body().scrollTop = body().scrollHeight;
  }

  async function send(text) {
    if (busy || !text.trim()) return;
    busy = true;
    $("#chat-send").disabled = true;
    $("#chat-chips").style.display = "none";
    addMsg("user", text);
    history.push({ role: "user", content: text });
    const pending = addMsg("bot", "분석 중…", true);
    try {
      const res = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: text, history: history.slice(0, -1) }),
      });
      if (!res.ok) throw new Error(`서버 오류 ${res.status}`);
      const data = await res.json();
      pending.remove();
      addMsg("bot", data.reply || "(응답 없음)");
      addTools(data.tool_trace);
      history.push({ role: "assistant", content: data.reply || "" });
    } catch (err) {
      pending.remove();
      addMsg("bot", `응답을 받지 못했습니다. (${err.message})\n로컬이면 서버가 켜져 있는지, 배포본이면 잠시 후 다시 시도해 주세요.`);
    } finally {
      busy = false;
      $("#chat-send").disabled = false;
      $("#chat-input").focus();
    }
  }

  async function refreshStatus() {
    const dot = $("#chat-status-dot");
    const label = $("#chat-status-label");
    try {
      const res = await fetch("/api/chat/health");
      const d = await res.json();
      if (d.llm_connected) {
        dot.className = "status-dot on";
        label.textContent = `LLM 연결됨 · ${d.model}`;
      } else {
        dot.className = "status-dot off";
        label.textContent = "규칙기반 모드 (API 키 미설정)";
      }
    } catch {
      dot.className = "status-dot off";
      label.textContent = "오프라인";
    }
  }

  function mount() {
    const fab = document.createElement("button");
    fab.className = "chat-fab";
    fab.id = "chat-fab";
    fab.innerHTML = `<span class="fab-mark">AI</span> 어시스턴트`;

    const drawer = document.createElement("aside");
    drawer.className = "chat-drawer";
    drawer.id = "chat-drawer";
    drawer.setAttribute("aria-label", "AI 어시스턴트");
    drawer.innerHTML = `
      <div class="chat-head">
        <div><div class="title">AI 어시스턴트</div>
          <div class="sub"><span id="chat-status-dot" class="status-dot off"></span><span id="chat-status-label">상태 확인 중…</span></div></div>
        <button class="chat-close" id="chat-close" aria-label="닫기">×</button>
      </div>
      <div class="chat-body" id="chat-body">
        <div class="chat-msg bot">안녕하세요. 주조 공정 품질 관제를 돕는 AI 어시스턴트입니다. 현황·위험 원인·임계값 비용·불량 예측·조치 후보를 물어보세요.\n\n제안·조치는 모두 관리자 승인 전 후보이며, 설비를 제어하거나 알림을 자동 발송하지 않습니다.</div>
      </div>
      <div class="chat-chips" id="chat-chips"></div>
      <div class="chat-input">
        <textarea id="chat-input" rows="1" placeholder="질문을 입력하세요 (Enter 전송, Shift+Enter 줄바꿈)"></textarea>
        <button class="chat-send" id="chat-send">전송</button>
      </div>
      <div class="chat-note">답변은 도구 결과에 근거하며, 통계적 관계는 인과가 아닙니다.</div>`;

    document.body.appendChild(fab);
    document.body.appendChild(drawer);

    const chips = $("#chat-chips");
    SAMPLES.forEach((s) => {
      const b = document.createElement("button");
      b.className = "chat-chip";
      b.textContent = s;
      b.onclick = () => send(s);
      chips.appendChild(b);
    });

    const open = () => { drawer.classList.add("open"); fab.classList.add("hidden"); $("#chat-input").focus(); refreshStatus(); };
    const close = () => { drawer.classList.remove("open"); fab.classList.remove("hidden"); };
    fab.onclick = open;
    $("#chat-close").onclick = close;

    const input = $("#chat-input");
    input.addEventListener("input", () => { input.style.height = "auto"; input.style.height = Math.min(input.scrollHeight, 120) + "px"; });
    input.addEventListener("keydown", (e) => {
      if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); const v = input.value; input.value = ""; input.style.height = "auto"; send(v); }
    });
    $("#chat-send").onclick = () => { const v = input.value; input.value = ""; input.style.height = "auto"; send(v); };
  }

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", mount);
  else mount();
})();
