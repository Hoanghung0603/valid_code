const state={health:null,selected:null,runId:null,kind:"chunks",offset:0,limit:20,totals:{docs:0,chunks:0}};
const $=id=>document.getElementById(id);
const escapeText=value=>String(value??"");

function notify(message,type="info"){
  const toast=$("toast");
  toast.textContent=message;
  toast.className=`toast ${type==="error"?"error":""} show`;
  clearTimeout(window.toastTimer);
  window.toastTimer=setTimeout(()=>toast.className="toast",3600);
}

async function api(url,options){const response=await fetch(url,options);if(!response.ok){let detail=response.statusText;try{detail=(await response.json()).detail||detail}catch{}throw new Error(detail)}return response.json()}

function metric(value,label){const node=document.createElement("div");node.className="metric";const b=document.createElement("b");b.textContent=value;const s=document.createElement("span");s.textContent=label;node.append(b,s);return node}

async function init(){
  try{
    state.health=await api("/api/health");
    $("release-id").textContent=state.health.release_id+" · healthy";
    const m=$("metrics");m.replaceChildren(
      metric(state.health.public_metrics.final_score.toFixed(4),"Public Final · evaluator mới"),
      metric(state.health.public_metrics.docs_f2_macro.toFixed(4),"Documents F2-macro"),
      metric(state.health.public_metrics.chunks_f2_macro.toFixed(4),"Chunks F2-macro"),
      metric(state.health.counts.index_chunks.toLocaleString("vi-VN"),"Chunks trong active index")
    );
    const f=$("fingerprints");for(const [key,value] of Object.entries(state.health.fingerprints)){const row=document.createElement("div");const s=document.createElement("span");s.textContent=key.toUpperCase()+" SHA256";const c=document.createElement("code");c.textContent=value;row.append(s,c);f.append(row)}
    const queryData=await loadQueries();
    if(new URLSearchParams(location.search).get("demo")==="1"&&queryData.items.length){
      selectQuery(queryData.items[0],document.querySelector(".query-option"));
      await runReplay();
      const first=await api(`/api/runs/${state.runId}/results?kind=chunks&limit=1`);
      if(first.items.length)await loadEvidence(first.items[0].chunk_ref);
    }
  }catch(error){$("release-id").textContent="Release unavailable";notify(error.message,"error")}
}

async function loadQueries(){const search=$("query-search").value.trim();const data=await api(`/api/queries?search=${encodeURIComponent(search)}&limit=50`);const list=$("query-list");list.replaceChildren();for(const item of data.items){const button=document.createElement("button");button.className="query-option";button.type="button";button.setAttribute("aria-pressed","false");const b=document.createElement("b");b.textContent=`QUERY ${item.id}`;const span=document.createElement("span");span.textContent=item.query;button.append(b,span);button.onclick=()=>selectQuery(item,button);list.append(button)}if(!data.items.length){list.textContent="Không tìm thấy query khớp."}return data}

function selectQuery(item,button){document.querySelectorAll(".query-option").forEach(n=>{n.classList.remove("active");n.setAttribute("aria-pressed","false")});button.classList.add("active");button.setAttribute("aria-pressed","true");state.selected=item;$("selected-id").textContent=`QUERY ${item.id}`;$("selected-text").textContent=item.query;$("run-button").disabled=false;document.dispatchEvent(new CustomEvent("r2ai:query-selected",{detail:{id:item.id}}))}

async function runReplay(){if(!state.selected)return;$("run-button").disabled=true;$("run-button").setAttribute("aria-busy","true");$("run-button").textContent="Đang mở cache…";try{const run=await api("/api/runs",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({mode:"replay",query_id:state.selected.id,query:state.selected.query})});state.runId=run.run_id;state.offset=0;$("export-button").disabled=false;$("result-summary").textContent=`Query ${run.query_id} · replay hoàn tất`;await refreshCounts();await loadResults();notify(`Replay query ${run.query_id} đã sẵn sàng`)}catch(error){notify(error.message,"error")}finally{$("run-button").disabled=false;$("run-button").removeAttribute("aria-busy");$("run-button").textContent="Chạy replay"}}

async function refreshCounts(){for(const kind of ["docs","chunks"]){const data=await api(`/api/runs/${state.runId}/results?kind=${kind}&limit=1`);state.totals[kind]=data.total}$("doc-count").textContent=state.totals.docs;$("chunk-count").textContent=state.totals.chunks}

async function loadResults(){if(!state.runId)return;if(state.kind==="trace"){renderTrace();return}const data=await api(`/api/runs/${state.runId}/results?kind=${state.kind}&offset=${state.offset}&limit=${state.limit}`);const list=$("result-list");list.className="result-list";list.replaceChildren();for(const item of data.items){const card=document.createElement("article");card.className="result-card";const meta=document.createElement("div");meta.className="result-meta";const left=document.createElement("span");left.textContent=`#${item.ordinal+1}`;const right=document.createElement("span");right.textContent=`DOC ${item.doc_id}`;meta.append(left,right);card.append(meta);if(state.kind==="chunks"){card.tabIndex=0;card.setAttribute("role","button");card.setAttribute("aria-label",`Mở provenance cho đoạn ${item.ordinal+1}, tài liệu ${item.doc_id}`);const p=document.createElement("p");p.textContent=item.chunk_text;card.append(p);card.onclick=()=>loadEvidence(item.chunk_ref);card.onkeydown=e=>{if(e.key==="Enter"||e.key===" "){e.preventDefault();loadEvidence(item.chunk_ref)}}}else{const p=document.createElement("p");p.textContent="Tài liệu thuộc corpus BTC";const url=document.createElement("div");url.className="doc-url";url.textContent=item.url||"URL không có trong mapping";card.append(p,url)}list.append(card)}renderPagination(data.total)}

function renderPagination(total){const pages=Math.max(1,Math.ceil(total/state.limit)),page=Math.floor(state.offset/state.limit)+1;$("pagination").classList.toggle("hidden",total<=state.limit);$("page-label").textContent=`${page} / ${pages}`;$("prev").disabled=state.offset===0;$("next").disabled=state.offset+state.limit>=total}

function renderTrace(){const list=$("result-list");list.className="result-list";list.replaceChildren();const grid=document.createElement("div");grid.className="trace-grid";const rows={"Mode":"replay · cached inference","Release":state.health.release_id,"Submission SHA256":state.health.fingerprints.submission,"Predictions SHA256":state.health.fingerprints.predictions,"Live inference":"Unavailable trong review release","Private score":"Chưa công bố"};for(const [label,value] of Object.entries(rows)){const r=document.createElement("div");r.className="trace-row";const s=document.createElement("span");s.textContent=label;const c=document.createElement("code");c.textContent=value;r.append(s,c);grid.append(r)}list.append(grid);$("pagination").classList.add("hidden")}

async function loadEvidence(ref){const target=$("evidence");target.className="";target.textContent="Đang resolve source snapshot…";try{const item=await api(`/api/runs/${state.runId}/evidence/${ref}`);target.replaceChildren();const meta=document.createElement("div");meta.className="evidence-meta";for(const [text,cls] of [[`DOC ${item.doc_id}`,""],[item.resolved?"EXACT RESOLVED":"UNRESOLVED",item.resolved?"resolved":"unresolved"],[item.source_part||"no source part",""]]){const p=document.createElement("span");p.className=`pill ${cls}`;p.textContent=text;meta.append(p)}target.append(meta);if(item.url&&/^https?:\/\//i.test(item.url)){const a=document.createElement("a");a.className="source-link";a.href=item.url;a.target="_blank";a.rel="noopener noreferrer";a.textContent=item.url;target.append(a)}const box=document.createElement("div");box.className="evidence-text";if(item.resolved){const before=document.createTextNode(item.context.slice(0,item.highlight_start));const mark=document.createElement("mark");mark.textContent=item.context.slice(item.highlight_start,item.highlight_end);const after=document.createTextNode(item.context.slice(item.highlight_end));box.append(before,mark,after)}else{box.textContent=item.chunk_text+"\n\n["+item.reason+"]"}target.append(box);const hash=document.createElement("p");hash.className="hash";hash.textContent=`chunk sha256: ${item.text_sha256}`+(item.source_sha256?`\nsource sha256: ${item.source_sha256}`:"");target.append(hash)}catch(error){target.textContent=error.message;notify(error.message,"error")}}

$("query-search").addEventListener("input",()=>{clearTimeout(window.queryTimer);window.queryTimer=setTimeout(loadQueries,220)});$("run-button").onclick=runReplay;$("export-button").onclick=()=>{if(state.runId)location.href=`/api/runs/${state.runId}/export`};document.querySelectorAll(".tab").forEach(tab=>tab.onclick=()=>{document.querySelectorAll(".tab").forEach(t=>{t.classList.remove("active");t.setAttribute("aria-selected","false")});tab.classList.add("active");tab.setAttribute("aria-selected","true");state.kind=tab.dataset.kind;state.offset=0;loadResults()});$("prev").onclick=()=>{state.offset=Math.max(0,state.offset-state.limit);loadResults()};$("next").onclick=()=>{state.offset+=state.limit;loadResults()};
init();
