"use strict";
const el = (id) => document.getElementById(id);
const example = {
  q1: {type: "enum", description: "What is 2 + 3?\nA. 4\nB. 5\nC. 6", choices: ["A", "B", "C"]},
  q2: {type: "enum", description: "Which planet is closest to the Sun?\nA. Venus\nB. Mars\nC. Mercury", choices: ["A", "B", "C"]}
};
function loadExample() { el("schema-input").value = JSON.stringify(example, null, 2); }
function seconds(value) { return Number.isFinite(value) ? `${value.toFixed(3)} s` : "—"; }
async function health() {
  try {
    const response = await fetch("/health");
    if (!response.ok) throw new Error("服务不可用");
    const status = await response.json();
    el("health").textContent = status.model_loaded ? "● 模型已就绪" : "● 已连接 · 模型待加载";
  } catch { el("health").textContent = "○ 服务未连接"; }
}
function renderAnswers(answers) {
  el("answers").replaceChildren();
  for (const [name, answer] of Object.entries(answers)) {
    const card = document.createElement("article"); card.className = "answer";
    const heading = document.createElement("div"); heading.className = "answer-heading";
    const title = document.createElement("strong"); title.textContent = name;
    const winner = document.createElement("span"); winner.className = "winner";
    winner.textContent = `${answer.value} · γ ${(answer.gamma * 100).toFixed(1)}%`;
    heading.append(title, winner); card.append(heading);
    if (answer.provenance?.candidate_token_ids) {
      const p = answer.provenance;
      const detail = document.createElement("details");
      const summary = document.createElement("summary"); summary.textContent = "本字段 logits 的实际取值位置";
      const content = document.createElement("pre");
      content.textContent = `方法：${p.method}\n文本解码器：${p.decoder_layer_count} 层，最后 block 索引 ${p.last_decoder_layer_index}\n最终 RMSNorm → ${p.module}\n选位后 output.logits[${p.batch_index}, ${p.selected_logit_index}, ${JSON.stringify(p.candidate_token_ids)}]\n等价于 full_logits[${p.batch_index}, ${p.absolute_position}, ids]（0-based）\n本行后缀：${JSON.stringify(p.slot_text)}\n后缀 token IDs：${JSON.stringify(p.slot_token_ids)}\n候选顺序：${JSON.stringify(p.candidate_labels)}\n直接读取的 logits：${JSON.stringify(answer.logits)}\n温度 T = ${p.temperature}`;
      detail.append(summary, content); card.append(detail);
    }
    for (const [label, probability] of Object.entries(answer.probabilities)) {
      const row = document.createElement("div"); row.className = "probability";
      const text = document.createElement("span"); text.textContent = label;
      const bar = document.createElement("progress"); bar.max = 1; bar.value = probability;
      bar.setAttribute("aria-label", `${name} / ${label}`);
      const value = document.createElement("span"); value.textContent = `${(probability * 100).toFixed(2)}%`;
      row.append(text, bar, value); card.append(row);
    }
    el("answers").append(card);
  }
}
el("example").addEventListener("click", loadExample);
el("decision-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  el("error").hidden = true;
  el("answers").replaceChildren();
  el("raw-details").hidden = true;
  el("raw").textContent = "";
  el("calibration").textContent = "等待本次响应的校准状态。";
  for (const id of ["roundtrip", "decision-time", "setup-time"]) el(id).textContent = "—";
  let schema;
  try {
    schema = JSON.parse(el("schema-input").value);
    if (!schema || Array.isArray(schema) || typeof schema !== "object") throw new Error("Schema 必须是 JSON 对象。");
    if (Object.keys(schema).length < 1 || Object.keys(schema).length > 10) throw new Error("请输入 1–10 个字段。");
  } catch (error) {
    el("error").textContent = `输入错误：${error.message}`; el("error").hidden = false;
    el("result-status").textContent = "输入无效"; return;
  }
  el("submit").disabled = true; el("example").disabled = true;
  el("output-panel").setAttribute("aria-busy", "true");
  el("result-status").textContent = "运行中…";
  const started = performance.now();
  const timer = setInterval(() => { el("elapsed").textContent = `已等待 ${((performance.now() - started) / 1000).toFixed(1)} s`; }, 100);
  try {
    const response = await fetch("/v1/calibrated-schema", {
      method: "POST", headers: {"Content-Type": "application/json"},
      body: JSON.stringify({schema, state: el("state-input").value || "Answer each question using its listed options."})
    });
    const text = await response.text();
    el("roundtrip").textContent = seconds((performance.now() - started) / 1000);
    let result;
    try { result = JSON.parse(text); } catch { throw new Error(`服务返回 HTTP ${response.status}，未得到 JSON。请查看服务日志。`); }
    if (!response.ok) throw new Error(typeof result.detail === "string" ? result.detail : JSON.stringify(result.detail));
    renderAnswers(result.answers);
    el("decision-time").textContent = seconds(result.timing?.decision_seconds);
    el("setup-time").textContent = seconds(result.timing?.model_setup_seconds);
    el("calibration").textContent = result.calibration_fitted ? "已加载温度校准参数；分布外正确率仍不保证。" : "未校准：γ 是选中选项的归一化概率，不是经验证的正确率。";
    el("raw").textContent = JSON.stringify(result, null, 2); el("raw-details").hidden = false;
    el("result-status").textContent = `${Object.keys(result.answers).length} 个字段 · ${result.usage.forward_calls} 次 forward`;
  } catch (error) {
    el("error").textContent = error.message; el("error").hidden = false;
    el("result-status").textContent = "请求失败";
  } finally {
    clearInterval(timer); el("elapsed").textContent = `请求结束 · ${((performance.now() - started) / 1000).toFixed(3)} s`;
    el("submit").disabled = false; el("example").disabled = false;
    el("output-panel").setAttribute("aria-busy", "false"); health();
  }
});
loadExample(); health();
