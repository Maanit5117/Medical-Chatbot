/**
 * Medical RAG Chatbot - Frontend Application
 */

document.addEventListener("DOMContentLoaded", () => {
  // DOM Elements
  const chatMessages = document.getElementById("chatMessages");
  const userInput = document.getElementById("userInput");
  const sendBtn = document.getElementById("sendBtn");
  const clearChatBtn = document.getElementById("clearChatBtn");
  const charCounter = document.getElementById("charCounter");
  const menuToggle = document.getElementById("menuToggle");
  const sidebar = document.getElementById("sidebar");
  const infoModalBtn = document.getElementById("infoModalBtn");
  const infoModal = document.getElementById("infoModal");
  const closeModalBtn = document.getElementById("closeModalBtn");
  const queryChips = document.querySelectorAll(".query-chip, .suggestion-card");
  const toast = document.getElementById("toast");

  let isGenerating = false;
  let synth = window.speechSynthesis;
  let currentUtterance = null;

  // Initialize
  userInput.focus();
  updateCharCounter();
  checkHealth();

  // Scroll smoothly to bottom
  function scrollToBottom() {
    chatMessages.scrollTop = chatMessages.scrollHeight;
    requestAnimationFrame(() => {
      chatMessages.scrollTop = chatMessages.scrollHeight;
    });
  }

  // Health check to verify backend / Pinecone status
  async function checkHealth() {
    const statusBadge = document.querySelector(".status-badge");
    if (!statusBadge) return;
    try {
      const res = await fetch("/api/health");
      const data = await res.json();
      if (res.ok && data.pinecone === "online") {
        statusBadge.classList.remove("offline");
        statusBadge.querySelector("span").textContent = "Online";
      } else {
        statusBadge.classList.add("offline");
        statusBadge.querySelector("span").textContent = "Offline";
      }
    } catch (_) {
      statusBadge.classList.add("offline");
      statusBadge.querySelector("span").textContent = "Offline";
    }
  }

  // Auto-resize input
  userInput.addEventListener("input", () => {
    userInput.style.height = "auto";
    userInput.style.height = `${Math.min(userInput.scrollHeight, 140)}px`;
    updateCharCounter();
    sendBtn.disabled = userInput.value.trim().length === 0 || isGenerating;
  });

  // Handle enter key (Enter sends, Shift+Enter new line)
  userInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      if (!isGenerating && userInput.value.trim().length > 0) {
        handleSubmit();
      }
    }
  });

  // Send button click
  sendBtn.addEventListener("click", () => {
    if (!isGenerating && userInput.value.trim().length > 0) {
      handleSubmit();
    }
  });

  // Quick query suggestions
  queryChips.forEach((chip) => {
    chip.addEventListener("click", () => {
      const query = chip.getAttribute("data-query") || chip.querySelector("h4")?.textContent;
      if (query && !isGenerating) {
        userInput.value = query.replace(/[🩺🔬🫁🦴]/g, "").trim();
        userInput.style.height = "auto";
        updateCharCounter();
        handleSubmit();
        if (window.innerWidth <= 900) {
          sidebar.classList.remove("open");
        }
      }
    });
  });

  // Clear Chat
  if (clearChatBtn) {
    clearChatBtn.addEventListener("click", () => {
      if (confirm("Are you sure you want to clear the conversation?")) {
        resetToWelcome();
        showToast("Chat history cleared");
      }
    });
  }

  // Sidebar toggle for mobile
  if (menuToggle && sidebar) {
    menuToggle.addEventListener("click", () => {
      sidebar.classList.toggle("open");
    });
  }

  // Modal open/close
  if (infoModalBtn && infoModal) {
    infoModalBtn.addEventListener("click", () => {
      infoModal.classList.add("open");
    });
  }

  if (closeModalBtn && infoModal) {
    closeModalBtn.addEventListener("click", () => {
      infoModal.classList.remove("open");
    });
  }

  if (infoModal) {
    infoModal.addEventListener("click", (e) => {
      if (e.target === infoModal) {
        infoModal.classList.remove("open");
      }
    });
  }

  function updateCharCounter() {
    if (charCounter) {
      charCounter.textContent = `${userInput.value.length} / 2000`;
    }
  }

  // Reset to Welcome Screen
  function resetToWelcome() {
    chatMessages.innerHTML = `
      <div class="welcome-container" id="welcomeScreen">
        <div class="welcome-icon-box">
          <svg width="36" height="36" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2">
            <path d="M22 12h-4l-3 9L9 3l-3 9H2"/>
          </svg>
        </div>
        <h2>MediRAG</h2>
        <p>Ask clinical and health questions. Answers drawn from the <strong>Gale Encyclopedia of Medicine (2nd edition)</strong>.</p>
        
        <div class="welcome-suggestions">
          <div class="suggestion-card" data-query="What is acne and what causes it?">
            <h4>🩺 Acne Causes & Symptoms</h4>
            <p>Understand the dermatological pathology and triggers</p>
          </div>
          <div class="suggestion-card" data-query="What is gigantism and how is it diagnosed?">
            <h4>🔬 Gigantism Details</h4>
            <p>Endocrine disorder symptoms and pituitary context</p>
          </div>
          <div class="suggestion-card" data-query="What is byssinosis and who is at risk?">
            <h4>🫁 Byssinosis Overview</h4>
            <p>Occupational lung disease in textile workers</p>
          </div>
          <div class="suggestion-card" data-query="What is bursitis and how is it treated?">
            <h4>🦴 Bursitis Therapy</h4>
            <p>Joint inflammation, anti-inflammatory care & prevention</p>
          </div>
        </div>
      </div>
    `;

    // Re-attach listeners to new suggestion cards
    document.querySelectorAll(".welcome-suggestions .suggestion-card").forEach((card) => {
      card.addEventListener("click", () => {
        const query = card.getAttribute("data-query");
        if (query) {
          userInput.value = query;
          handleSubmit();
        }
      });
    });
  }

  // Main submission pipeline
  async function handleSubmit() {
    const text = userInput.value.trim();
    if (!text || isGenerating) return;

    // Remove welcome screen if present
    const welcome = document.getElementById("welcomeScreen");
    if (welcome) {
      welcome.remove();
    }

    // Render User Message
    appendMessage({
      sender: "user",
      text: text,
      time: getCurrentTime(),
    });

    // Reset input
    userInput.value = "";
    userInput.style.height = "auto";
    updateCharCounter();
    sendBtn.disabled = true;
    isGenerating = true;

    // Append Typing Indicator
    const typingElement = showTypingIndicator();
    scrollToBottom();

    try {
      // Call backend API
      const response = await fetchChatResponse(text);
      typingElement.remove();

      appendMessage({
        sender: "bot",
        text: response.answer || "No response received.",
        sources: response.sources || [],
        time: getCurrentTime(),
      });
    } catch (err) {
      console.error("Chat error:", err);
      typingElement.remove();

      let friendlyMessage = "Something went wrong on our side. Please try again.";
      const errStr = (err.errorBody || err.message || "").toString();

      if (err.name === "AbortError" || err.isTimeout) {
        friendlyMessage = "⏱️ Request timed out after 60 seconds. The server took too long to respond. Please try again.";
      } else if (err.status === 504 || errStr.includes("TIMEOUT_ERROR") || errStr.toLowerCase().includes("timed out") || errStr.toLowerCase().includes("timeout")) {
        friendlyMessage = "⏱️ The AI service timed out. Please try again.";
      } else if (err.isNetworkError || err.name === "TypeError" || (!err.status && err.message.includes("fetch"))) {
        friendlyMessage = "Can't reach the server. Is it running?";
      } else if (err.status === 429 || errStr.includes("RESOURCE_EXHAUSTED") || errStr.includes("429") || errStr.toLowerCase().includes("quota")) {
        friendlyMessage = "Daily AI usage limit reached. Please try again later.";
      } else if (err.status >= 500) {
        if (errStr.includes("RESOURCE_EXHAUSTED") || errStr.includes("429")) {
          friendlyMessage = "Daily AI usage limit reached. Please try again later.";
        } else {
          friendlyMessage = "Something went wrong on our side. Please try again.";
        }
      }

      appendMessage({
        sender: "bot",
        text: friendlyMessage,
        sources: [],
        time: getCurrentTime(),
        isError: true,
      });
    } finally {
      isGenerating = false;
      sendBtn.disabled = userInput.value.trim().length === 0;
      scrollToBottom();
    }
  }

  // Fetch logic with fallback endpoints and 60s AbortController
  async function fetchChatResponse(query) {
    let res;
    let endpoint = "/api/chat";
    const controller = new AbortController();
    const timeoutId = setTimeout(() => {
      controller.abort();
    }, 60000); // 60s client-side timeout

    try {
      try {
        res = await fetch(endpoint, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ message: query }),
          signal: controller.signal,
        });
      } catch (fetchErr) {
        if (fetchErr.name === "AbortError") {
          throw fetchErr;
        }
        // If /api/chat fails (network), try fallback to /get
        endpoint = `/get?msg=${encodeURIComponent(query)}`;
        res = await fetch(endpoint, {
          method: "GET",
          signal: controller.signal,
        });
      }
    } catch (err) {
      clearTimeout(timeoutId);
      if (err.name === "AbortError") {
        const timeoutErr = new Error("Request timed out after 60 seconds.");
        timeoutErr.name = "AbortError";
        timeoutErr.isTimeout = true;
        throw timeoutErr;
      }
      const netErr = new Error("NETWORK_FAILURE");
      netErr.isNetworkError = true;
      throw netErr;
    } finally {
      clearTimeout(timeoutId);
    }

    if (!res.ok) {
      let errorBody = "";
      try {
        const json = await res.json();
        errorBody = json.message || json.error || json.detail || JSON.stringify(json);
      } catch (_) {
        try {
          errorBody = await res.text();
        } catch (_) {}
      }
      const httpErr = new Error(errorBody || res.statusText);
      httpErr.status = res.status;
      httpErr.errorBody = errorBody;
      throw httpErr;
    }

    const contentType = res.headers.get("content-type");
    if (contentType && contentType.includes("application/json")) {
      return await res.json();
    } else {
      // If server returned plain text
      const plainText = await res.text();
      return { answer: plainText, sources: [] };
    }
  }

  // Append a message bubble to the chat container
  function appendMessage({ sender, text, sources = [], time, isError = false }) {
    const row = document.createElement("div");
    row.className = `message-row ${sender}`;

    const avatar = document.createElement("div");
    avatar.className = `avatar ${sender}`;
    if (sender === "user") {
      avatar.innerHTML = `<svg width="18" height="18" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24"><path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"/><circle cx="12" cy="7" r="4"/></svg>`;
    } else {
      avatar.innerHTML = `<svg width="18" height="18" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24"><path d="M22 12h-4l-3 9L9 3l-3 9H2"/></svg>`;
    }

    const wrapper = document.createElement("div");
    wrapper.className = "message-wrapper";

    const bubble = document.createElement("div");
    bubble.className = "message-bubble";
    if (isError) {
      bubble.classList.add("error-bubble");
      bubble.style.borderColor = "#fca5a5";
      bubble.style.background = "#fff5f5";
      bubble.style.color = "#991b1b";
    }

    bubble.innerHTML = formatMarkdown(text);
    wrapper.appendChild(bubble);

    // If bot and has sources, create sources accordion
    if (sender === "bot" && sources && sources.length > 0) {
      const sourcesBox = createSourcesAccordion(sources);
      wrapper.appendChild(sourcesBox);
    }

    // Bot Action Bar (Copy, TTS)
    if (sender === "bot" && !isError) {
      const actions = document.createElement("div");
      actions.className = "message-actions";

      // Copy Button
      const copyBtn = document.createElement("button");
      copyBtn.className = "action-btn";
      copyBtn.innerHTML = `<svg width="13" height="13" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24"><rect width="14" height="14" x="8" y="8" rx="2" ry="2"/><path d="M4 16c-1.1 0-2-.9-2-2V4c0-1.1.9-2 2-2h10c1.1 0 2 .9 2 2"/></svg> <span>Copy</span>`;
      copyBtn.addEventListener("click", () => {
        navigator.clipboard.writeText(text).then(() => {
          showToast("Answer copied to clipboard!");
          copyBtn.querySelector("span").textContent = "Copied!";
          setTimeout(() => {
            copyBtn.querySelector("span").textContent = "Copy";
          }, 2000);
        });
      });
      actions.appendChild(copyBtn);

      // TTS Read Aloud Button
      if ("speechSynthesis" in window) {
        const speechBtn = document.createElement("button");
        speechBtn.className = "action-btn";
        speechBtn.innerHTML = `<svg width="13" height="13" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24"><polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5"/><path d="M15.54 8.46a5 5 0 0 1 0 7.07"/><path d="M19.07 4.93a10 10 0 0 1 0 14.14"/></svg> <span>Listen</span>`;
        speechBtn.addEventListener("click", () => {
          toggleSpeech(text, speechBtn);
        });
        actions.appendChild(speechBtn);
      }

      wrapper.appendChild(actions);
    }

    // Timestamp
    const timeElem = document.createElement("div");
    timeElem.className = "message-time";
    timeElem.textContent = time;
    wrapper.appendChild(timeElem);

    row.appendChild(avatar);
    row.appendChild(wrapper);

    chatMessages.appendChild(row);
    scrollToBottom();
  }

  // Create RAG Sources Accordion
  function createSourcesAccordion(sources) {
    const container = document.createElement("div");
    container.className = "sources-container";

    const header = document.createElement("div");
    header.className = "sources-header";
    header.innerHTML = `
      <div class="sources-header-left">
        <svg width="14" height="14" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24">
          <path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20"/>
          <path d="M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2z"/>
        </svg>
        <span>Retrieved References (${sources.length} excerpts)</span>
      </div>
      <svg class="sources-toggle-icon" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24">
        <polyline points="6 9 12 15 18 9"/>
      </svg>
    `;

    const list = document.createElement("div");
    list.className = "sources-list";
    list.style.display = "none";

    sources.forEach((item, idx) => {
      const sourceCard = document.createElement("div");
      sourceCard.className = "source-item";

      const sourcePath = item.source || item.metadata?.source || "data/Medical_book.pdf";
      const content = item.content || item.page_content || item.text || JSON.stringify(item);

      sourceCard.innerHTML = `
        <div class="source-item-meta">
          <svg width="12" height="12" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24">
            <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/>
            <polyline points="14 2 14 8 20 8"/>
          </svg>
          <span>Source #${idx + 1}: ${escapeHtml(sourcePath)}</span>
        </div>
        <div class="source-item-content">${escapeHtml(content)}</div>
      `;
      list.appendChild(sourceCard);
    });

    header.addEventListener("click", () => {
      const isOpen = list.style.display === "flex";
      list.style.display = isOpen ? "none" : "flex";
      header.classList.toggle("open", !isOpen);
    });

    container.appendChild(header);
    container.appendChild(list);
    return container;
  }

  // Show Typing Indicator
  function showTypingIndicator() {
    const row = document.createElement("div");
    row.className = "message-row bot";
    row.id = "typingRow";

    row.innerHTML = `
      <div class="avatar bot">
        <svg width="18" height="18" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24">
          <path d="M22 12h-4l-3 9L9 3l-3 9H2"/>
        </svg>
      </div>
      <div class="message-wrapper">
        <div class="typing-indicator">
          <div class="typing-dot"></div>
          <div class="typing-dot"></div>
          <div class="typing-dot"></div>
          <span class="typing-text">Searching Pinecone & synthesizing answer...</span>
        </div>
      </div>
    `;

    chatMessages.appendChild(row);
    scrollToBottom();
    return row;
  }

  // Text to Speech
  function toggleSpeech(text, btn) {
    if (synth.speaking) {
      synth.cancel();
      btn.querySelector("span").textContent = "Listen";
      btn.classList.remove("active");
      return;
    }

    // Clean markdown before speaking
    const cleanText = text.replace(/[*#_`>]/g, "");
    currentUtterance = new SpeechSynthesisUtterance(cleanText);
    currentUtterance.rate = 1.0;
    currentUtterance.pitch = 1.0;

    currentUtterance.onstart = () => {
      btn.querySelector("span").textContent = "Stop";
      btn.classList.add("active");
    };

    currentUtterance.onend = () => {
      btn.querySelector("span").textContent = "Listen";
      btn.classList.remove("active");
    };

    currentUtterance.onerror = () => {
      btn.querySelector("span").textContent = "Listen";
      btn.classList.remove("active");
    };

    synth.speak(currentUtterance);
  }

  // Helpers
  function getCurrentTime() {
    const now = new Date();
    return now.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
  }

  function showToast(msg) {
    if (!toast) return;
    toast.textContent = msg;
    toast.classList.add("show");
    setTimeout(() => {
      toast.classList.remove("show");
    }, 2800);
  }

  function escapeHtml(str) {
    if (!str) return "";
    return String(str)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#039;");
  }

  // Lightweight safe markdown formatting
  function formatMarkdown(text) {
    if (!text) return "";
    let html = escapeHtml(text);

    // Code blocks ```code```
    html = html.replace(/```([\s\S]*?)```/g, (match, code) => {
      return `<pre><code>${code.trim()}</code></pre>`;
    });

    // Inline code `code`
    html = html.replace(/`([^`]+)`/g, "<code>$1</code>");

    // Bold **text**
    html = html.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");

    // Italic *text*
    html = html.replace(/\*([^*]+)\*/g, "<em>$1</em>");

    // Headers ### Title
    html = html.replace(/^### (.*$)/gim, "<h3>$1</h3>");
    html = html.replace(/^## (.*$)/gim, "<h2>$1</h2>");
    html = html.replace(/^# (.*$)/gim, "<h1>$1</h1>");

    // Line breaks and list items
    const lines = html.split("\n");
    let inList = false;
    let result = [];

    for (let line of lines) {
      const trimmed = line.trim();
      if (trimmed.startsWith("- ") || trimmed.startsWith("* ")) {
        if (!inList) {
          result.push("<ul>");
          inList = true;
        }
        result.push(`<li>${trimmed.substring(2)}</li>`);
      } else if (/^\d+\.\s/.test(trimmed)) {
        if (!inList) {
          result.push("<ol>");
          inList = "ol";
        }
        result.push(`<li>${trimmed.replace(/^\d+\.\s/, "")}</li>`);
      } else {
        if (inList) {
          result.push(inList === "ol" ? "</ol>" : "</ul>");
          inList = false;
        }
        if (trimmed.length > 0) {
          result.push(`<p>${trimmed}</p>`);
        }
      }
    }

    if (inList) {
      result.push(inList === "ol" ? "</ol>" : "</ul>");
    }

    return result.join("");
  }
});
