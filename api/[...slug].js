/**
 * 东盟跨境灾害心理干预应急语言服务平台 —— 云端后端（Vercel Serverless Functions）
 * 环境变量（Vercel 控制台 Settings → Environment Variables）：
 *   BIGMODEL_API_KEY   智谱 BigModel 密钥
 *   DEEPSEEK_API_KEY   DeepSeek 密钥
 * 可选：
 *   SITE_BASE          静态站点根地址
 *   ALLOW_ORIGIN       允许的前端来源，默认 *
 *   RAG_MIN_SCORE      拒答阈值，默认 0.30
 */

const DIMS = 512;
const TOP_K = 4;
const DEFAULT_SITE = "https://hpeppa.github.io/asean-psych-support-platform/";

let vectorCache = null;
let glossaryCache = null;

function siteBase() {
  return (process.env.SITE_BASE || DEFAULT_SITE).replace(/\/?$/, "/");
}

function corsHeaders() {
  const origin = process.env.ALLOW_ORIGIN || "*";
  return {
    "Access-Control-Allow-Origin": origin,
    "Access-Control-Allow-Methods": "GET,POST,OPTIONS",
    "Access-Control-Allow-Headers": "Content-Type",
    "Access-Control-Max-Age": "86400"
  };
}

function b64ToInt8(b64) {
  const bin = atob(b64);
  const out = new Int8Array(bin.length);
  for (let i = 0; i < bin.length; i++) {
    const b = bin.charCodeAt(i);
    out[i] = b > 127 ? b - 256 : b;
  }
  return out;
}

async function loadVectors() {
  if (vectorCache) return vectorCache;
  const res = await fetch(siteBase() + "kb_vectors.json");
  if (!res.ok) throw new Error("无法加载 kb_vectors.json：" + res.status);
  const data = await res.json();
  vectorCache = data.items.map(function (item) {
    const arr = b64ToInt8(item.v);
    let norm = 0;
    for (let i = 0; i < arr.length; i++) {
      const x = arr[i] * item.s;
      norm += x * x;
    }
    return { meta: item, vec: arr, scale: item.s, norm: Math.sqrt(norm) || 1 };
  });
  return vectorCache;
}

async function embedQuery(text) {
  const res = await fetch("https://open.bigmodel.cn/api/paas/v4/embeddings", {
    method: "POST",
    headers: {
      "Authorization": "Bearer " + process.env.BIGMODEL_API_KEY,
      "Content-Type": "application/json"
    },
    body: JSON.stringify({ model: "embedding-3", input: [text], dimensions: DIMS })
  });
  if (!res.ok) throw new Error("BigModel 调用失败：" + res.status + " " + (await res.text()).slice(0, 160));
  const data = await res.json();
  return data.data[0].embedding;
}

function rank(items, query, filterType, limit) {
  let qn = 0;
  for (let i = 0; i < query.length; i++) qn += query[i] * query[i];
  qn = Math.sqrt(qn) || 1;
  const scored = [];
  for (let k = 0; k < items.length; k++) {
    const row = items[k];
    if (filterType && filterType.indexOf(row.meta.type) < 0) continue;
    let dot = 0;
    const n = Math.min(row.vec.length, query.length);
    for (let i = 0; i < n; i++) dot += row.vec[i] * row.scale * query[i];
    scored.push({ meta: row.meta, score: dot / (row.norm * qn) });
  }
  scored.sort(function (a, b) { return b.score - a.score; });
  return scored.slice(0, limit || TOP_K);
}

async function loadGlossary() {
  if (glossaryCache) return glossaryCache;
  const [gl, terms] = await Promise.all([
    fetch(siteBase() + "glossary.json").then(function (r) { return r.json(); }),
    fetch(siteBase() + "insarag_terms.json").then(function (r) { return r.json(); })
  ]);
  glossaryCache = (terms || []).concat(gl || []);
  return glossaryCache;
}

const LANG_FIELD = { zh: "term_zh", en: "term_en", th: "term_th", vi: "term_vi", ms: "term_ms" };

function normalizeLang(v) {
  const s = String(v || "").toLowerCase();
  if (s.indexOf("en") === 0) return "en";
  if (s.indexOf("th") === 0) return "th";
  if (s.indexOf("vi") === 0) return "vi";
  if (s.indexOf("ms") === 0 || s.indexOf("may") === 0) return "ms";
  return "zh";
}

function matchTerms(pool, text, sourceLang, targetLang) {
  const sf = LANG_FIELD[sourceLang] || "term_zh";
  const tf = LANG_FIELD[targetLang] || "term_en";
  const hits = [];
  for (const item of pool) {
    const src = String(item[sf] || "").trim();
    if (!src || src.length < 2) continue;
    const idx = text.indexOf(src);
    if (idx < 0) continue;
    hits.push({
      source_term: src,
      target_term: String(item[tf] || ""),
      domain: item.domain || "",
      review_status: item.review_status || "待审核",
      position: idx,
      length: src.length
    });
  }
  hits.sort(function (a, b) { return b.length - a.length; });
  const used = [];
  const kept = [];
  for (const hit of hits) {
    const overlap = used.some(function (u) {
      return hit.position < u.position + u.length && u.position < hit.position + hit.length;
    });
    if (overlap) continue;
    used.push(hit);
    kept.push(hit);
  }
  return kept.sort(function (a, b) { return a.position - b.position; });
}

const SYS_CHAT = "你是东盟跨境灾害心理干预应急语言服务平台的问答助手，专长包括 PFA 心理急救、MHPSS、创伤应激、哀伤辅导、儿童与家庭心理支持、救援人员心理关怀。只能依据给定的知识库片段作答，不得编造；片段不足以回答时要明确说明。回答中引用片段时用 [1][2] 标注编号。";

async function deepseekStream(messages) {
  return fetch("https://api.deepseek.com/chat/completions", {
    method: "POST",
    headers: {
      "Authorization": "Bearer " + process.env.DEEPSEEK_API_KEY,
      "Content-Type": "application/json"
    },
    body: JSON.stringify({
      model: process.env.DEEPSEEK_MODEL || "deepseek-chat",
      messages: messages,
      stream: true,
      temperature: 0.3
    })
  });
}

async function deepseekOnce(messages, maxTokens) {
  const res = await fetch("https://api.deepseek.com/chat/completions", {
    method: "POST",
    headers: {
      "Authorization": "Bearer " + process.env.DEEPSEEK_API_KEY,
      "Content-Type": "application/json"
    },
    body: JSON.stringify({
      model: process.env.DEEPSEEK_MODEL || "deepseek-chat",
      messages: messages,
      stream: false,
      temperature: 0.2,
      max_tokens: maxTokens || 1200
    })
  });
  if (!res.ok) throw new Error("DeepSeek 调用失败：" + res.status + " " + (await res.text()).slice(0, 160));
  const data = await res.json();
  return data.choices[0].message.content;
}

async function handleStatus() {
  let items = [];
  try { items = await loadVectors(); } catch (e) { items = []; }
  const count = function (t) { return items.filter(function (x) { return x.meta.type === t; }).length; };
  return {
    ok: true,
    index_ready: items.length > 0,
    chunk_count: items.length,
    corpus_count: count("corpus"),
    glossary_count: (glossaryCache ? glossaryCache.length : 0) || 507,
    parallel_count: count("corpus"),
    foreign_count: count("foreign"),
    news_count: count("news"),
    pfa_count: count("pfa"),
    embedding_mode: "bigmodel",
    embedding_model: "embedding-3",
    embedding_dim: DIMS,
    llm_mode: "deepseek",
    llm_model: process.env.DEEPSEEK_MODEL || "deepseek-chat",
    min_score: Number(process.env.RAG_MIN_SCORE || 0.30),
    corpus_vectorized: true,
    mode: "cloud"
  };
}

function sendSSE(res, obj) {
  res.write("data: " + JSON.stringify(obj) + "\n\n");
}

async function handleChat(body, res) {
  const question = String(body.question || "").trim();
  if (!question) {
    res.status(400).json({ ok: false, error: "问题不能为空" });
    return;
  }

  const minScore = Number(process.env.RAG_MIN_SCORE || 0.30);
  const items = await loadVectors();
  const query = await embedQuery(question);
  const hits = rank(items, query, null, TOP_K);
  const best = hits.length ? hits[0].score : 0;
  const refused = best < minScore;

  const status = {
    chunk_count: items.length,
    corpus_count: items.filter(function (x) { return x.meta.type === "corpus"; }).length,
    embedding_mode: "bigmodel",
    embedding_model: "embedding-3",
    llm_mode: "deepseek",
    llm_model: process.env.DEEPSEEK_MODEL || "deepseek-chat",
    min_score: minScore
  };

  res.writeHead(200, Object.assign({
    "Content-Type": "text/event-stream; charset=utf-8",
    "Cache-Control": "no-cache",
    "X-Accel-Buffering": "no"
  }, corsHeaders()));

  if (refused) {
    sendSSE(res, Object.assign({ type: "retrieval", sources: [], refused: true }, status));
    sendSSE(res, { type: "done", mode: "refusal", elapsed_ms: 0 });
    res.end();
    return;
  }

  const sources = hits.map(function (h, i) {
    const m = h.meta;
    return {
      index: i + 1, type: m.type, keyword: m.keyword, title: m.title,
      summary: m.summary, publish_date: m.publish_date, url: m.url,
      source_name: m.source_name, score: h.score
    };
  });

  const context = hits.map(function (h, i) {
    const m = h.meta;
    return "[" + (i + 1) + "] 标题：" + m.title + "\n摘要：" + m.summary +
      (m.publish_date ? "\n日期：" + m.publish_date : "") +
      (m.source_name ? "\n来源：" + m.source_name : "");
  }).join("\n\n");

  const langHint = { zh: "简体中文", en: "English", th: "ภาษาไทย", vi: "Tiếng Việt", ms: "Bahasa Melayu" };
  const lang = normalizeLang(body.lang);
  const upstream = await deepseekStream([
    { role: "system", content: SYS_CHAT + "\n请使用" + (langHint[lang] || "简体中文") + "回答。" },
    { role: "user", content: "知识库片段：\n" + context + "\n\n用户问题：" + question }
  ]);

  if (!upstream.ok || !upstream.body) {
    const text = await upstream.text();
    sendSSE(res, Object.assign({ type: "retrieval", sources: sources, refused: false }, status));
    sendSSE(res, { type: "error", message: "生成失败：" + upstream.status + " " + text.slice(0, 120) });
    res.end();
    return;
  }

  const reader = upstream.body.getReader();
  const decoder = new TextDecoder();
  const started = Date.now();
  let buffer = "";
  let sentRetrieval = false;

  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const parts = buffer.split("\n");
      buffer = parts.pop() || "";
      for (const line of parts) {
        const trimmed = line.trim();
        if (!trimmed.startsWith("data:")) continue;
        const payload = trimmed.slice(5).trim();
        if (payload === "[DONE]") continue;
        try {
          const obj = JSON.parse(payload);
          const delta = obj.choices && obj.choices[0] && obj.choices[0].delta;
          const content = delta && delta.content;
          if (content) {
            if (!sentRetrieval) {
              sentRetrieval = true;
              sendSSE(res, Object.assign({ type: "retrieval", sources: sources, refused: false }, status));
            }
            sendSSE(res, { type: "delta", content: content });
          }
        } catch (e) { /* 忽略分片不完整的 JSON */ }
      }
    }
    if (!sentRetrieval) {
      sendSSE(res, Object.assign({ type: "retrieval", sources: sources, refused: false }, status));
    }
    sendSSE(res, { type: "done", mode: "deepseek", elapsed_ms: Date.now() - started });
  } catch (err) {
    sendSSE(res, { type: "error", message: String(err).slice(0, 160) });
  } finally {
    res.end();
  }
}

async function handleTranslate(body, res) {
  const started = Date.now();
  const text = String(body.text || "").trim();
  if (!text) {
    res.status(400).json({ ok: false, error: "原文不能为空" });
    return;
  }
  const sourceLang = normalizeLang(body.source);
  const targetLang = normalizeLang(body.target);

  const [items, pool] = await Promise.all([loadVectors(), loadGlossary()]);
  const query = await embedQuery(text);

  const refs = rank(items, query, ["corpus"], 3);
  const news = rank(items, query, ["news", "foreign", "pfa", "glossary"], 3);
  const terms = matchTerms(pool, text, sourceLang, targetLang);

  const langName = { zh: "简体中文", en: "English", th: "ภาษาไทย", vi: "Tiếng Việt", ms: "Bahasa Melayu" };
  const termBlock = terms.length
    ? terms.map(function (t) { return "- " + t.source_term + " → " + t.target_term; }).join("\n")
    : "（未命中术语库）";
  const corpusBlock = refs.length
    ? refs.map(function (r) { return "- 源：" + (r.meta.summary || "").split(" ｜ ")[0] + "\n  译：" + (r.meta.summary || "").split(" ｜ ")[1]; }).join("\n")
    : "（无语料参考）";
  const newsBlock = news.length
    ? news.map(function (r) { return "- " + r.meta.title; }).join("\n")
    : "（无相关资讯）";

  const prompt = [
    "你是专业的" + (langName[sourceLang] || "") + "到" + (langName[targetLang] || "") + "翻译引擎，领域为东盟灾害救援与灾后心理干预（PFA、MHPSS、创伤应激、哀伤辅导）。",
    "要求：1) 必须严格使用下方术语表中的对应译法；2) 参考平行语料的句式与措辞；3) 只输出译文，不要解释、不要加引号。",
    "",
    "术语表：",
    termBlock,
    "",
    "平行语料参考：",
    corpusBlock,
    "",
    "相关资讯（供术语与背景参考）：",
    newsBlock,
    "",
    "待翻译原文：",
    text
  ].join("\n");

  const translation = await deepseekOnce([
    { role: "system", content: "你是严谨的灾害与心理干预领域专业译者。" },
    { role: "user", content: prompt }
  ]);

  res.setHeader("Content-Type", "application/json; charset=utf-8");
  Object.keys(corsHeaders()).forEach(function (k) { res.setHeader(k, corsHeaders()[k]); });
  res.status(200).json({
    ok: true,
    translation: translation.trim(),
    matches: terms,
    references: refs.map(function (r) {
      const parts = String(r.meta.summary || "").split(" ｜ ");
      return {
        source_text: parts[0] || r.meta.title,
        target_text: parts[1] || "",
        lang_pair: r.meta.lang_pair || "",
        domain: r.meta.keyword || "",
        score: r.score,
        source: r.meta.source_name || ""
      };
    }),
    news_sources: news.map(function (r) {
      return {
        title: r.meta.title, summary: r.meta.summary, keyword: r.meta.keyword,
        source_name: r.meta.source_name, publish_date: r.meta.publish_date,
        url: r.meta.url, type: r.meta.type, score: r.score
      };
    }),
    elapsed_ms: Date.now() - started,
    mode: "deepseek",
    model: process.env.DEEPSEEK_MODEL || "deepseek-chat",
    source_lang: sourceLang,
    target_lang: targetLang
  });
}

export default async function handler(req, res) {
  // CORS headers for all responses
  const cors = corsHeaders();
  Object.keys(cors).forEach(function (k) { res.setHeader(k, cors[k]); });

  if (req.method === "OPTIONS") {
    res.status(204).end();
    return;
  }

  const slug = (req.query.slug || []).join("/");
  const path = "/" + slug;

  try {
    if (path === "/api/status" && req.method === "GET") {
      res.status(200).json(await handleStatus());
      return;
    }
    if (path === "/api/chat" && req.method === "POST") {
      await handleChat(req.body, res);
      return;
    }
    if (path === "/api/translate" && req.method === "POST") {
      await handleTranslate(req.body, res);
      return;
    }
    if (path === "/api/rebuild") {
      res.status(200).json({ ok: true, message: "云端向量由构建脚本预计算，更新数据后重新运行 build_vectors.py 并重新上传 kb_vectors.json。" });
      return;
    }
    if (path === "/" || path === "/health" || slug.length === 0) {
      res.status(200).json({ ok: true, service: "asean-psych-support-platform", routes: ["/api/status", "/api/chat", "/api/translate"] });
      return;
    }
    res.status(404).json({ ok: false, error: "接口不存在" });
  } catch (err) {
    res.status(500).json({ ok: false, error: String(err).slice(0, 200) });
  }
}
