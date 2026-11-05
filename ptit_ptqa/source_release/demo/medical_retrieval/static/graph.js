/* Experimental corpus mention graph; never modifies audited replay output. */
(() => {
  const byId = id => document.getElementById(id);
  const ns = "http://www.w3.org/2000/svg";
  let available = false;
  let sequence = 0;

  function svg(tag, attrs = {}) {
    const node = document.createElementNS(ns, tag);
    for (const [name, value] of Object.entries(attrs)) node.setAttribute(name, String(value));
    return node;
  }

  function label(value, max = 34) {
    return value.length > max ? value.slice(0, max - 1) + "…" : value;
  }

  function draw(data) {
    const visual = byId("graph-visual");
    const details = byId("graph-details");
    visual.replaceChildren();
    details.replaceChildren();
    const concepts = data.concepts;
    const docs = data.documents;
    if (!concepts.length) { visual.textContent = "Không có cạnh để hiển thị trong cache cho query này."; return; }
    visual.classList.remove("graph-visual-empty");
    const height = Math.max(380, concepts.length * 58 + 60, docs.length * 52 + 60);
    const canvas = svg("svg", {viewBox:`0 0 1000 ${height}`, width:"100%", height,
      "aria-hidden":"true", preserveAspectRatio:"xMidYMid meet"});
    const queryY = height / 2;
    const conceptY = new Map(concepts.map((c, i) => [c.id, height / (concepts.length + 1) * (i + 1)]));
    const docY = docs.map((_, i) => height / (docs.length + 1) * (i + 1));
    for (const concept of concepts) {
      canvas.append(svg("path", {d:`M 216 ${queryY} C 275 ${queryY}, 280 ${conceptY.get(concept.id)}, 338 ${conceptY.get(concept.id)}`,
        fill:"none", stroke:"#8cbdb7", "stroke-width":1.5}));
    }
    docs.forEach((doc, i) => {
      for (const edge of doc.evidence) {
        const y = conceptY.get(edge.concept_id);
        if (y === undefined) continue;
        canvas.append(svg("path", {d:`M 558 ${y} C 625 ${y}, 625 ${docY[i]}, 682 ${docY[i]}`,
          fill:"none", stroke:"#c0d4d1", "stroke-width":1.5}));
      }
    });
    function node(x, y, width, text, type, detail) {
      const group = svg("g", {class:`kg-node kg-${type}`});
      group.append(svg("rect", {x, y:y-19, width, height:38, rx:9}));
      const caption = svg("text", {x:x+12, y:y+5});
      caption.textContent = label(text, type === "doc" ? 27 : 31);
      group.append(caption);
      const title = svg("title"); title.textContent = detail || text; group.append(title);
      canvas.append(group);
    }
    node(18, queryY, 198, `QUERY ${data.query_id}`, "query", data.query);
    concepts.forEach(c => node(338, conceptY.get(c.id), 220, c.label || c.id.replaceAll("_", " "), "concept",
      c.chapter ? `${c.label || c.id} · ICD chapter ${c.chapter}` : `${c.label || c.id} · ${c.kind}`));
    docs.forEach((d, i) => node(682, docY[i], 290,
      `DOC ${d.doc_id} · ${data.mode === "corpus_concepts" ? `${d.overlap} term` : `chunk #${d.chunk_rank || "—"}`}`,
      "doc", d.url || `Cached document ${d.doc_id}`));
    visual.append(canvas);

    const heading = document.createElement("h3");
    heading.textContent = docs.length ? "Cạnh có thể kiểm tra" : "Chưa có tài liệu khớp trong graph";
    details.append(heading);
    for (const doc of docs) {
      const article = document.createElement("article"); article.className = "graph-detail";
      const strong = document.createElement("strong"); strong.textContent = `DOC ${doc.doc_id}`;
      const path = document.createElement("span");
      path.textContent = doc.evidence.map(e => e.source === "replay_rank"
        ? `Replay chunk #${doc.chunk_rank || "—"}` : `${e.alias} · ${e.source === "corpus_mention" ? `corpus offset ${e.normalized_offset}` : "chunk"}`).join(" · ");
      article.append(strong, path);
      if (/^https?:\/\//i.test(doc.url)) {
        const link = document.createElement("a"); link.href = doc.url;
        link.target = "_blank"; link.rel = "noopener noreferrer";
        link.textContent = "Mở URL nguồn"; article.append(link);
      }
      const excerpt = doc.evidence.find(e => e.snippet)?.snippet;
      if (excerpt) {
        const quote = document.createElement("p");
        quote.className = "graph-snippet";
        quote.textContent = excerpt;
        article.append(quote);
      }
      details.append(article);
    }
  }

  async function loadGraph(queryId) {
    if (!available) return;
    const current = ++sequence;
    byId("graph-status").textContent = `Đang dựng graph cho query ${queryId}…`;
    try {
      const response = await fetch(`/api/graph/queries/${encodeURIComponent(queryId)}?limit=8`);
      if (!response.ok) throw new Error(`Graph API: ${response.status}`);
      const data = await response.json();
      if (current !== sequence) return;
      draw(data);
      const basis = data.mode === "corpus_concepts" ? "Thuật ngữ trong corpus" :
        data.edge_basis === "normalized_chunk_overlap" ? "Khớp từ trong chunk replay" : "Đường truy hồi trong cache";
      byId("graph-status").textContent = `Query ${queryId} · ${basis}`;
      byId("graph-counts").textContent = `${data.concepts.length} nút trung gian · ${data.documents.length} tài liệu minh họa`;
      byId("graph-visual").setAttribute("aria-label",
        `Graph query ${queryId} nối ${data.concepts.length} nút trung gian với ${data.documents.length} tài liệu; ${basis}`);
    } catch (error) {
      if (current !== sequence) return;
      byId("graph-status").textContent = "Graph tạm không khả dụng";
      byId("graph-visual").textContent = error.message;
    }
  }

  document.addEventListener("r2ai:query-selected", event => loadGraph(event.detail.id));
  fetch("/api/graph/status").then(response => response.json()).then(status => {
    available = !!status.available;
    byId("graph-status").textContent = status.concept_graph_available
      ? "Concept graph + replay fallback · experimental" : "Replay evidence graph · experimental";
    byId("graph-counts").textContent = status.concept_graph_available
      ? `${status.concepts} khái niệm curated · ${status.indexed_docs.toLocaleString("vi-VN")} tài liệu` : "Hoạt động không cần SQLite sidecar";
    if (available && window.location.search.includes("demo=1")) loadGraph(1);
  }).catch(() => { byId("graph-status").textContent = "Graph không khả dụng"; });
})();
