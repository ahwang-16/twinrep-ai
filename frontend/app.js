let activeTab = "chat";

function getToken() {
  return sessionStorage.getItem("twinrep_manager_token") || "";
}

function authHeaders(extra = {}) {
  const token = getToken();
  return token ? { ...extra, Authorization: `Bearer ${token}` } : extra;
}

function updateAuthUI() {
  const token = getToken();
  const personaSelect = document.getElementById("personaSelect");
  const authChip = document.getElementById("authHeaderStatus");

  if (token) {
    personaSelect.disabled = false;
    authChip.innerHTML = `Manager: admin <button type="button" onclick="logoutManager()">Logout</button>`;
  } else {
    personaSelect.disabled = true;
    authChip.textContent = "Customer View";
  }
}

function switchTab(tabName, btn) {
  activeTab = tabName;
  document.querySelectorAll(".tab-panel").forEach(p => p.classList.remove("active"));
  document.querySelectorAll(".tab-btn").forEach(b => b.classList.remove("active"));
  btn.classList.add("active");

  if ((tabName === "kb" || tabName === "manager") && !getToken()) {
    document.getElementById("managerLoginGate").classList.add("active");
    return;
  }

  document.getElementById(`tab-${tabName}`).classList.add("active");
  if (tabName === "manager") loadTickets();
  if (tabName === "kb") refreshHealth();
}

async function handleManagerLogin(e) {
  e.preventDefault();
  const username = document.getElementById("loginUsername").value;
  const password = document.getElementById("loginPassword").value;
  const errEl = document.getElementById("loginError");
  errEl.textContent = "";

  const res = await fetch("/api/auth/login", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ username, password }),
  });

  if (!res.ok) {
    const err = await res.json();
    errEl.textContent = err.detail || "Login failed.";
    return;
  }

  const data = await res.json();
  sessionStorage.setItem("twinrep_manager_token", data.access_token);
  updateAuthUI();

  document.getElementById("managerLoginGate").classList.remove("active");
  document.getElementById(`tab-${activeTab}`).classList.add("active");
  if (activeTab === "manager") loadTickets();
}

function logoutManager() {
  sessionStorage.removeItem("twinrep_manager_token");
  updateAuthUI();
  const chatBtn = document.querySelector(".tab-btn");
  switchTab("chat", chatBtn);
}

async function refreshHealth() {
  try {
    const res = await fetch("/api/health");
    const data = await res.json();

    const dbEl = document.getElementById("healthDb");
    const faissEl = document.getElementById("healthFaiss");
    const llmEl = document.getElementById("healthLlm");

    const dbOk = data.mysql === "connected";
    dbEl.textContent = `MySQL: ${data.mysql}`;
    dbEl.className = `health-pill ${dbOk ? "ok" : "err"}`;

    const chunks = data.faiss?.total_chunks ?? 0;
    const files = data.faiss?.indexed_files ?? [];
    faissEl.textContent = `FAISS: ${chunks} chunks`;
    faissEl.className = `health-pill ${chunks > 0 ? "ok" : "err"}`;

    const filesCountEl = document.getElementById("indexedFilesCount");
    const filesListEl = document.getElementById("indexedFilesList");
    if (filesCountEl) filesCountEl.textContent = `${files.length} files indexed`;
    if (filesListEl) {
      filesListEl.textContent = files.length
        ? `Active Indexed Files: ${files.join(", ")}`
        : "No manual files indexed yet.";
    }

    const llmOk = data.llm?.status === "online";
    const scaleText = data.llm?.lora_scale !== null && data.llm?.lora_scale !== undefined
      ? ` (LoRA=${data.llm.lora_scale})`
      : "";
    llmEl.textContent = `LLM: ${data.llm?.status || "offline"}${scaleText}`;
    llmEl.className = `health-pill ${llmOk ? "ok" : "err"}`;

    if (data.llm?.lora_scale !== null && data.llm?.lora_scale !== undefined) {
      document.getElementById("personaSelect").value = Number(data.llm.lora_scale).toFixed(1);
    }
  } catch (e) {
    console.error("Health check error:", e);
  }
}

async function updatePersonaScale(scaleVal) {
  const res = await fetch("/api/persona/scale", {
    method: "POST",
    headers: authHeaders({ "Content-Type": "application/json" }),
    body: JSON.stringify({ scale: parseFloat(scaleVal) }),
  });
  if (!res.ok) {
    const err = await res.json();
    alert(err.detail || "Failed to update LoRA scale.");
  }
  refreshHealth();
}

async function sendQuestion(e) {
  e.preventDefault();
  const nameInput = document.getElementById("customerName");
  const qInput = document.getElementById("questionInput");
  const sendBtn = document.getElementById("sendBtn");
  const chatBox = document.getElementById("chatMessages");

  const question = qInput.value.trim();
  if (!question) return;

  chatBox.innerHTML += `<div class="msg user"><strong>${nameInput.value}:</strong>\n${question}</div>`;
  qInput.value = "";
  sendBtn.disabled = true;
  sendBtn.textContent = "Thinking...";

  try {
    const res = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ customer_name: nameInput.value, question }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "Server error");

    document.getElementById("lookupTicketId").value = data.ticket_id;
    document.getElementById("lookupStatusText").textContent =
      `Ticket #${data.ticket_id} logged (${data.status.replace("_", " ")}). Click Check Manager Verification for updates.`;

    const sourcesText = (data.sources || [])
      .map(s => `${s.source} (${Math.round(s.similarity * 100)}%)`)
      .join(", ") || "None (below similarity threshold)";

    chatBox.innerHTML += `
      <div class="msg bot">
        <strong>TwinRep Agent (Ticket #${data.ticket_id} — AI Draft):</strong>\n${data.ai_draft}
        <div class="sources-pill">RAG Sources Referenced: ${sourcesText}</div>
      </div>`;
  } catch (err) {
    chatBox.innerHTML += `<div class="msg bot">Error contacting TwinRep server: ${err.message}</div>`;
  } finally {
    sendBtn.disabled = false;
    sendBtn.textContent = "Send Inquiry";
    chatBox.scrollTop = chatBox.scrollHeight;
  }
}

async function checkCustomerTicket() {
  const idVal = document.getElementById("lookupTicketId").value;
  const banner = document.getElementById("verifiedTicketBanner");
  if (!idVal) return alert("Enter a Ticket # first.");

  const res = await fetch(`/api/tickets/${idVal}/status`);
  if (!res.ok) {
    banner.style.display = "block";
    banner.className = "verified-banner pending";
    banner.textContent = `Ticket #${idVal} not found.`;
    return;
  }

  const t = await res.json();
  banner.style.display = "block";
  if (t.status === "pending_review") {
    banner.className = "verified-banner pending";
    banner.innerHTML = `<strong>Ticket #${t.id} Status: PENDING MANAGER REVIEW</strong>\nYour inquiry is awaiting final sign-off from a Senior After-Sales Manager.`;
  } else {
    banner.className = "verified-banner";
    banner.innerHTML = `<strong>Ticket #${t.id} Status: VERIFIED BY MANAGER (${t.status.toUpperCase()})</strong>\n<strong>Official Response:</strong>\n${t.final_response}`;
  }
}

async function uploadManual() {
  const fileInput = document.getElementById("manualFile");
  const statusEl = document.getElementById("uploadStatus");
  if (!fileInput.files.length) return alert("Select a PDF, TXT, or MD file first.");

  const formData = new FormData();
  formData.append("file", fileInput.files[0]);
  statusEl.textContent = "Checking hash, chunking, and embedding document...";

  const res = await fetch("/api/upload", {
    method: "POST",
    headers: authHeaders(),
    body: formData,
  });
  const data = await res.json();
  if (!res.ok) {
    statusEl.textContent = `Error: ${data.detail}`;
    return;
  }
  statusEl.textContent = data.message || `Processed "${data.filename}".`;
  refreshHealth();
}

async function syncFolder() {
  const statusEl = document.getElementById("uploadStatus");
  statusEl.textContent = "Scanning /data/manuals for new or updated files...";
  const res = await fetch("/api/ingest-folder", {
    method: "POST",
    headers: authHeaders(),
  });
  const data = await res.json();
  if (!res.ok) {
    statusEl.textContent = `Error: ${data.detail}`;
    return;
  }
  statusEl.textContent = `Sync complete: ${data.total_chunks_added} new chunks added (${data.files_skipped} unchanged files skipped).`;
  refreshHealth();
}

async function loadTickets() {
  const container = document.getElementById("ticketList");
  const res = await fetch("/api/tickets", { headers: authHeaders() });
  if (res.status === 401) {
    logoutManager();
    return;
  }
  const tickets = await res.json();

  if (!tickets.length) {
    container.innerHTML = "<p>No customer tickets yet. Ask a question in Tab 1 first.</p>";
    return;
  }

  container.innerHTML = tickets.map(t => {
    const isLocked = t.status === "approved" || t.status === "edited";
    const sourcesLabel = (t.sources || [])
      .map(s => `${s.source} (${Math.round((s.similarity || 0) * 100)}%)`)
      .join(", ") || "None";

    return `
      <div class="ticket-card">
        <div class="ticket-meta">
          <span><strong>Ticket #${t.id}</strong> — ${t.customer_name}</span>
          <span class="status-tag ${t.status}">${t.status.replace("_", " ")}</span>
        </div>
        <p><strong>Customer Question:</strong> ${t.question}</p>
        <p style="font-size:0.82rem; color:var(--amber);"><strong>Retrieved Context:</strong> ${sourcesLabel}</p>
        <label><strong>${isLocked ? "Final Verified Response (Locked):" : "AI Drafted Response (Editable):"}</strong></label>
        <textarea id="draft-${t.id}" ${isLocked ? "readonly" : ""}>${t.final_response || t.ai_draft}</textarea>
        <div class="ticket-actions">
          ${
            isLocked
              ? `<span class="lock-note">✓ Locked & Indexed in Live FAISS Knowledge Base</span>`
              : `<button class="approve" onclick="approveTicket(${t.id})">Approve & Index</button>
                 <button onclick="editTicket(${t.id})">Edit & Teach AI</button>`
          }
        </div>
      </div>
    `;
  }).join("");
}

async function approveTicket(id) {
  const res = await fetch(`/api/tickets/${id}/approve`, {
    method: "POST",
    headers: authHeaders(),
  });
  if (!res.ok) {
    const err = await res.json();
    alert(err.detail || "Could not approve ticket.");
  }
  loadTickets();
  refreshHealth();
}

async function editTicket(id) {
  const edited_response = document.getElementById(`draft-${id}`).value;
  const res = await fetch(`/api/tickets/${id}/edit`, {
    method: "POST",
    headers: authHeaders({ "Content-Type": "application/json" }),
    body: JSON.stringify({ edited_response }),
  });
  if (!res.ok) {
    const err = await res.json();
    return alert(err.detail || "Could not save edited ticket.");
  }
  alert(`Ticket #${id} locked and vectorized into the live RAG Knowledge Base!`);
  loadTickets();
  refreshHealth();
}

// Initialize header status on page load
updateAuthUI();
refreshHealth();
