const byId = (id) => document.getElementById(id);
const model = byId("model");
const turns = byId("turns");
const message = byId("message");
const status = byId("status");
const send = byId("send");
let config;
let session;
let working = false;

async function request(path, body) {
  const response = await fetch("/api/" + path, body === undefined ? {} : {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  const value = await response.json();
  if (!response.ok) {
    const error = new Error(typeof value.detail === "string" ? value.detail : "Check your settings and try again.");
    error.status = response.status;
    throw error;
  }
  return value;
}

function updateSession(value) {
  session = value;
  localStorage.setItem("ncm-model", session.model);
  localStorage.setItem("ncm-session:" + session.model, session.id);
  turns.replaceChildren();
  if (!session.turns.length) {
    const empty = document.createElement("p");
    empty.className = "empty";
    empty.textContent = "Your session starts here.";
    turns.append(empty);
  }
  const task = config.models.find((item) => item.name === session.model).task;
  for (const item of session.turns) {
    for (const [kind, label] of [["user", "You"], ["assistant", task === "xsum" ? "Summary" : "NCM"]]) {
      const bubble = document.createElement("article");
      const title = document.createElement("small");
      bubble.className = "message " + kind;
      title.textContent = label;
      bubble.append(title, document.createTextNode(item[kind] || "∅"));
      turns.append(bubble);
    }
  }
  turns.scrollTop = turns.scrollHeight;
}

function entries(element, rows) {
  element.replaceChildren();
  for (const [label, value] of rows) {
    const term = document.createElement("dt");
    const detail = document.createElement("dd");
    term.textContent = label;
    detail.textContent = value;
    element.append(term, detail);
  }
}

const ms = (n) => n.toFixed(2) + " ms";
const mib = (n) => (n / 1048576).toFixed(1) + " MiB";

function showStats(stats) {
  byId("stats-empty").hidden = Boolean(stats);
  byId("detail").hidden = !stats;
  if (!stats) {
    entries(byId("stats"), []);
    return;
  }
  const times = stats.milliseconds;
  const transformer = Object.entries(times).filter(([name]) => name.startsWith("transformer_"));
  const rows = [
    ["Backend", stats.backend.toUpperCase() + " · " + stats.profile],
    ["Generation", ms(times.total)],
    ["Input embedding", ms(times.input_embedding)],
    ["Transformer", ms(transformer.reduce((total, [, time]) => total + time, 0))],
    ["Neural decoder", ms(times.neural_decoder)],
    ["Model load", ms(stats.model_load_ms)],
    ["Checkpoint", mib(stats.checkpoint_bytes)],
    ["Process RSS", mib(stats.rss_bytes)],
  ];
  if ("accelerator_allocated_bytes" in stats) {
    rows.push(["Accelerator tensors", mib(stats.accelerator_allocated_bytes)]);
  }
  entries(byId("stats"), rows);
  entries(byId("stages"), Object.entries(times).map(([name, time]) => [name.replaceAll("_", " "), ms(time)]));
}

async function selectModel(fresh = false) {
  const spec = config.models.find((item) => item.name === model.value);
  if (!spec) return;
  const dialogue = spec.task === "dialogue";
  byId("title").textContent = dialogue ? "Dialogue workspace" : "XSum research mode";
  byId("mode").textContent = dialogue
    ? "Replies come from the selected dialogue-adapted checkpoint."
    : "This checkpoint summarizes the current passage. Session history is saved separately.";
  byId("history").disabled = !dialogue;
  byId("context").max = spec.context_limit;
  byId("context").placeholder = String(spec.context_limit);
  byId("context").value = "";
  message.placeholder = dialogue ? "Write a message…" : "Paste a passage to summarize…";
  const previous = fresh ? null : localStorage.getItem("ncm-session:" + spec.name);
  if (previous) {
    try {
      const value = await request("sessions/" + previous);
      if (value.model === spec.name) return updateSession(value);
    } catch (error) {
      if (error.status !== 404) throw error;
    }
  }
  updateSession(await request("sessions", { model: spec.name }));
}

function controls() {
  for (const element of [send, byId("new"), byId("reset")]) element.disabled = working || !session;
  model.disabled = working || !config?.models.length;
}

async function run(action) {
  if (working) return;
  working = true;
  controls();
  status.textContent = "";
  try {
    await action();
  } catch (error) {
    if (error.status === 409 && session) {
      try { updateSession(await request("sessions/" + session.id)); } catch {}
    }
    status.textContent = error.message;
  } finally {
    working = false;
    controls();
  }
}

model.addEventListener("change", () => run(async () => {
  session = undefined;
  showStats();
  await selectModel();
}));
byId("new").addEventListener("click", () => run(async () => { await selectModel(true); showStats(); }));
byId("reset").addEventListener("click", () => run(async () => {
  updateSession(await request("sessions/" + session.id + "/reset", { revision: session.revision }));
  showStats();
}));
byId("composer").addEventListener("submit", (event) => {
  event.preventDefault();
  run(async () => {
    if (!session || !message.value.trim()) return;
    const payload = {
      text: message.value,
      revision: session.revision,
      backend: byId("backend").value,
      profile: byId("profile").value,
      history_turns: Number(byId("history").value || 0),
    };
    for (const [field, id] of [["context_limit", "context"], ...["temperature", "top_k", "top_p", "cfg_scale"].map((key) => [key, key])]) {
      const input = byId(id);
      if (!input.reportValidity()) return;
      if (input.value !== "") payload[field] = Number(input.value);
    }
    status.textContent = "Generating…";
    const value = await request("sessions/" + session.id + "/messages", payload);
    updateSession(value.session);
    showStats(value.stats);
    if (message.value === payload.text) message.value = "";
    status.textContent = "";
    message.focus();
  });
});

run(async () => {
  config = await request("config");
  for (const item of config.models) model.add(new Option(item.name, item.name));
  for (const backend of config.backends) byId("backend").add(new Option(backend.toUpperCase(), backend));
  if (!config.models.length) {
    status.textContent = "No checkpoint found. Run ncm-download, then restart the server.";
    return;
  }
  const last = localStorage.getItem("ncm-model");
  if (config.models.some((item) => item.name === last)) model.value = last;
  await selectModel();
});
