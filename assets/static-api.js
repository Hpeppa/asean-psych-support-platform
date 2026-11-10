/*
 * 静态检索引擎（无后端版）
 * ------------------------------------------------------------
 * 部署到 GitHub Pages / Cloudflare Pages 等纯静态托管时，
 * 浏览器直接读取本地 JSON 数据文件，在页面内完成：
 *   1. /api/status      —— 统计信息
 *   2. /api/chat        —— 关键词/字符二元组余弦相似度检索 + 抽取式作答（SSE 流式）
 *   3. /api/translate   —— 术语命中 + 平行语料检索 + 相关新闻，辅助生成译文
 * 不读取任何密钥，不发起外部请求。
 */
(function () {
  "use strict";
  if (window.__KB_STATIC_API__) return;
  window.__KB_STATIC_API__ = true;

  var MIN_SCORE = 0.18;
  var DATA = {};

  function loadJson(name) {
    if (!DATA[name]) {
      DATA[name] = fetch(name, { cache: "no-cache" })
        .then(function (r) { return r.ok ? r.json() : []; })
        .catch(function () { return []; });
    }
    return DATA[name];
  }

  function loadAll() {
    return Promise.all([
      loadJson("insarag_news.json"),
      loadJson("foreign_news.json"),
      loadJson("glossary.json"),
      loadJson("pfa_knowledge.json"),
      loadJson("parallel.json"),
      loadJson("parallel_corpus_1to1.json"),
      loadJson("insarag_terms.json")
    ]).then(function (res) {
      return {
        news: res[0] || [],
        foreign: res[1] || [],
        glossary: res[2] || [],
        pfa: res[3] || [],
        parallel: res[4] || [],
        corpus1: res[5] || [],
        terms: res[6] || []
      };
    });
  }

  /* ---------------- 分词与相似度 ---------------- */
  function tokenize(text) {
    var s = String(text || "").toLowerCase();
    var tokens = [];
    var latin = s.match(/[a-z][a-z0-9\-]{1,}/g) || [];
    for (var i = 0; i < latin.length; i++) tokens.push(latin[i]);
    var cjk = s.replace(/[^\u4e00-\u9fa5]+/g, " ");
    var runs = cjk.split(" ");
    for (var r = 0; r < runs.length; r++) {
      var seg = runs[r];
      if (seg.length === 1) { tokens.push(seg); continue; }
      for (var j = 0; j < seg.length - 1; j++) tokens.push(seg.substr(j, 2));
    }
    return tokens;
  }

  function vectorOf(text) {
    var tokens = tokenize(text);
    var vec = Object.create(null);
    for (var i = 0; i < tokens.length; i++) {
      var t = tokens[i];
      vec[t] = (vec[t] || 0) + 1;
    }
    return vec;
  }

  function cosine(a, b) {
    var ka = Object.keys(a);
    if (!ka.length) return 0;
    var dot = 0;
    for (var i = 0; i < ka.length; i++) {
      var k = ka[i];
      if (b[k]) dot += a[k] * b[k];
    }
    if (!dot) return 0;
    var na = 0, nb = 0, kb = Object.keys(b);
    for (var x = 0; x < ka.length; x++) na += a[ka[x]] * a[ka[x]];
    for (var y = 0; y < kb.length; y++) nb += b[kb[y]] * b[kb[y]];
    if (!na || !nb) return 0;
    return dot / Math.sqrt(na * nb);
  }

  /* ---------------- 知识库切片（与后端同构） ---------------- */
  var KB_CACHE = null;

  function buildChunks(d) {
    if (KB_CACHE) return KB_CACHE;
    var chunks = [];
    (d.news || []).forEach(function (item, i) {
      var title = item.title || "";
      var summary = item.summary || "";
      if (!title && !summary) return;
      chunks.push({
        id: "news-" + (i + 1),
        text: [item.keyword, title, summary, item.publish_date].join(" "),
        meta: {
          keyword: item.keyword || "跨境救援", title: title, summary: summary,
          publish_date: item.publish_date || "", url: item.url || "",
          type: "news", source_name: "insarag_news.json"
        }
      });
    });
    (d.foreign || []).forEach(function (item) {
      chunks.push({
        id: item.id || ("foreign-" + chunks.length),
        text: [item.category, item.title_zh, item.summary_zh, item.publish_date].join(" "),
        meta: {
          keyword: item.category || "东盟灾害", title: item.title_zh || "",
          summary: item.summary_zh || "", publish_date: item.publish_date || "",
          url: item.url || "", type: "foreign",
          source_name: item.source_name || item.source_code || ""
        }
      });
    });
    var gl = d.glossary || [];
    if (gl.length) {
      var step = Math.max(1, Math.floor(gl.length / 45));
      for (var i = 0; i < gl.length && chunks.length < 9999; i += step) {
        var g = gl[i];
        if (!g) continue;
        if (chunks.filter(function (c) { return c.meta.type === "glossary"; }).length >= 45) break;
        chunks.push({
          id: g.id || ("term-" + chunks.length),
          text: [g.term_zh, g.term_en, g.definition_zh, g.domain, g.term_th, g.term_vi, g.term_ms].join(" "),
          meta: {
            keyword: g.domain || "术语",
            title: "术语：" + (g.term_zh || "") + "（" + (g.term_en || "") + "）",
            summary: (g.definition_zh || "") + " 多语种：" + [g.term_en, g.term_th, g.term_vi, g.term_ms].join(" / "),
            publish_date: "", url: g.source_url || "", type: "glossary",
            source_name: g.source || ""
          }
        });
      }
    }
    (d.pfa || []).forEach(function (item) {
      chunks.push({
        id: item.id || ("pfa-" + chunks.length),
        text: [item.title_zh, item.summary_zh, item.domain].join(" "),
        meta: {
          keyword: "心理干预知识", title: item.title_zh || "",
          summary: item.summary_zh || "", publish_date: "",
          url: item.url || "", type: "pfa", source_name: item.source_name || ""
        }
      });
    });
    KB_CACHE = chunks.map(function (c) {
      return { chunk: c, vec: vectorOf(c.text) };
    });
    return KB_CACHE;
  }

  /* ---------------- 翻译辅助 ---------------- */
  var LANG_FIELD = { zh: "term_zh", en: "term_en", th: "term_th", vi: "term_vi", ms: "term_ms" };
  var LANG_ORDER = ["zh", "en", "th", "vi", "ms"];

  function normalizeLang(v) {
    var s = String(v || "").toLowerCase();
    if (s.indexOf("en") === 0) return "en";
    if (s.indexOf("th") === 0) return "th";
    if (s.indexOf("vi") === 0) return "vi";
    if (s.indexOf("ms") === 0 || s.indexOf("may") === 0) return "ms";
    return "zh";
  }

  function matchTerms(text, sourceLang, targetLang) {
    if (!loadAllCache.__pool) {
      var pool = [];
      (loadAllCache.terms || []).forEach(function (t) { pool.push(t); });
      (loadAllCache.glossary || []).forEach(function (t) { pool.push(t); });
      loadAllCache.__pool = pool;
    }
    var sf = LANG_FIELD[sourceLang] || "term_zh";
    var tf = LANG_FIELD[targetLang] || "term_en";
    var hits = [];
    loadAllCache.__pool.forEach(function (item) {
      var src = String(item[sf] || "").trim();
      if (!src || src.length < 2) return;
      var idx = text.indexOf(src);
      if (idx < 0) return;
      hits.push({
        source_term: src,
        target_term: String(item[tf] || ""),
        domain: item.domain || "",
        review_status: item.review_status || "待审核",
        position: idx,
        length: src.length
      });
    });
    hits.sort(function (a, b) { return b.length - a.length; });
    var used = [];
    var kept = [];
    hits.forEach(function (hit) {
      var overlap = used.some(function (u) {
        return hit.position < u.position + u.length && u.position < hit.position + hit.length;
      });
      if (overlap) return;
      used.push(hit);
      kept.push(hit);
    });
    return kept.sort(function (a, b) { return a.position - b.position; });
  }

  function corpusOf(d) {
    var pool = [];
    (d.corpus1 || []).forEach(function (item) {
      item.__real = true;
      pool.push(item);
    });
    (d.parallel || []).forEach(function (item) { pool.push(item); });
    return pool;
  }

  function findReferences(d, text, sourceLang, targetLang) {
    var qv = vectorOf(text);
    var out = [];
    corpusOf(d).forEach(function (item) {
      var pair = String(item.lang_pair || "");
      var sl = item.source_lang || (pair.split("-")[0] || "zh");
      var tl = item.target_lang || (pair.split("-")[1] || "en");
      var same = normalizeLang(sl) === sourceLang && normalizeLang(tl) === targetLang;
      var rev = normalizeLang(sl) === targetLang && normalizeLang(tl) === sourceLang;
      if (!same && !rev) return;
      var base = same ? item.source_text : item.target_text;
      var score = cosine(qv, vectorOf(base));
      if (rev) score *= 0.9;
      out.push({
        source_text: item.source_text || "",
        target_text: item.target_text || "",
        lang_pair: pair || (sl + "-" + tl),
        domain: item.domain || "",
        score: score,
        source: item.source || (item.__real ? "用户语料库" : "演示语料"),
        real: !!item.__real
      });
    });
    out.sort(function (a, b) { return b.score - a.score; });
    return out.slice(0, 3);
  }

  function findNews(d, text) {
    var qv = vectorOf(text);
    var rows = [];
    (d.foreign || []).forEach(function (item) {
      rows.push({
        title: item.title_zh || "", summary: item.summary_zh || "",
        keyword: item.category || "", source_name: item.source_name || "",
        publish_date: item.publish_date || "", url: item.url || "", type: "foreign",
        score: cosine(qv, vectorOf([item.title_zh, item.summary_zh, item.category].join(" ")))
      });
    });
    (d.news || []).forEach(function (item) {
      rows.push({
        title: item.title || "", summary: item.summary || "",
        keyword: item.keyword || "", source_name: "insarag_news.json",
        publish_date: item.publish_date || "", url: item.url || "", type: "news",
        score: cosine(qv, vectorOf([item.title, item.summary, item.keyword].join(" ")))
      });
    });
    rows.sort(function (a, b) { return b.score - a.score; });
    return rows.slice(0, 3);
  }

  function translateText(text, sourceLang, targetLang, terms, refs) {
    var best = refs[0];
    if (best && best.score >= 0.45 && best.target_text) {
      return best.target_text;
    }
    var tf = LANG_FIELD[targetLang] || "term_en";
    var out = text;
    var applied = [];
    terms.forEach(function (hit) {
      if (!hit.target_term) return;
      out = out.split(hit.source_term).join(hit.target_term);
      applied.push(hit.source_term + " → " + hit.target_term);
    });
    if (!applied.length) {
      return out;
    }
    return out;
  }

  /* ---------------- 作答生成 ---------------- */
  var INTRO = {
    zh: "以下内容依据平台知识库检索生成（抽取式作答，未调用外部大模型）：",
    en: "Answer compiled from the on-site knowledge base (extractive mode, no external LLM):",
    th: "คำตอบรวบรวมจากฐานความรู้ของแพลตฟอร์ม (โหมดสกัดข้อความ ไม่เรียกใช้โมเดลภายนอก):",
    vi: "Câu trả lời được tổng hợp từ cơ sở tri thức của nền tảng (chế độ trích xuất, không gọi mô hình ngoài):",
    ms: "Jawapan dijana daripada pangkalan pengetahuan platform (mod petikan, tanpa model luar):"
  };

  function buildAnswer(hits, lang) {
    var lines = [INTRO[lang] || INTRO.zh, ""];
    hits.forEach(function (h, i) {
      var title = h.meta.title || h.meta.keyword || "";
      var summary = String(h.meta.summary || "").replace(/\s+/g, " ").trim();
      if (summary.length > 180) summary = summary.slice(0, 180) + "…";
      lines.push((i + 1) + ". " + title + "[" + (i + 1) + "]");
      if (summary) lines.push("   " + summary);
    });
    return lines.join("\n");
  }

  /* ---------------- 响应构造 ---------------- */
  function jsonResponse(obj) {
    return Promise.resolve(new Response(JSON.stringify(obj), {
      status: 200,
      headers: { "Content-Type": "application/json; charset=utf-8" }
    }));
  }

  function sseResponse(events) {
    var enc = new TextEncoder();
    var stream = new ReadableStream({
      start: function (controller) {
        var i = 0;
        function push() {
          if (i >= events.length) { controller.close(); return; }
          var ev = events[i++];
          controller.enqueue(enc.encode("data: " + JSON.stringify(ev) + "\n\n"));
          setTimeout(push, ev.type === "delta" ? 28 : 60);
        }
        push();
      }
    });
    return new Response(stream, {
      status: 200,
      headers: { "Content-Type": "text/event-stream; charset=utf-8", "Cache-Control": "no-cache" }
    });
  }

  var loadAllCache = {};

  function statusPayload(d) {
    var chunks = buildChunks(d);
    loadAllCache = d;
    return {
      ok: true,
      index_ready: true,
      chunk_count: chunks.length,
      corpus_count: (d.parallel || []).length + (d.corpus1 || []).length,
      glossary_count: (d.glossary || []).length + (d.terms || []).length,
      parallel_count: (d.parallel || []).length + (d.corpus1 || []).length,
      foreign_count: (d.foreign || []).length,
      news_count: (d.news || []).length,
      pfa_count: (d.pfa || []).length,
      embedding_mode: "static",
      embedding_model: "本地检索（字符二元组余弦）",
      embedding_dim: 0,
      llm_mode: "static",
      llm_model: "抽取式作答",
      min_score: MIN_SCORE,
      mode: "static",
      warning: ""
    };
  }

  function handleChat(body) {
    return loadAll().then(function (d) {
      loadAllCache = d;
      var question = String(body.question || "").trim();
      var lang = normalizeLang(body.lang);
      var chunks = buildChunks(d);
      var qv = vectorOf(question);
      var scored = chunks.map(function (row) {
        return { chunk: row.chunk, score: cosine(qv, row.vec) };
      });
      scored.sort(function (a, b) { return b.score - a.score; });
      var top = scored.slice(0, 4).filter(function (r) { return r.score > 0.02; });
      var refused = !top.length || top[0].score < MIN_SCORE;

      if (refused) {
        return sseResponse([
          {
            type: "retrieval", sources: [], refused: true, min_score: MIN_SCORE,
            chunk_count: chunks.length, corpus_count: (d.parallel || []).length,
            embedding_mode: "static", embedding_model: "本地检索（字符二元组余弦）",
            llm_mode: "static", llm_model: "抽取式作答"
          },
          { type: "done", mode: "refusal", elapsed_ms: 320 }
        ]);
      }

      var sources = top.map(function (r, i) {
        var m = r.chunk.meta;
        return {
          index: i + 1, type: m.type, keyword: m.keyword, title: m.title,
          summary: m.summary, publish_date: m.publish_date, url: m.url,
          source_name: m.source_name, score: r.score
        };
      });
      var answer = buildAnswer(top.map(function (r) { return r.chunk; }), lang);
      var events = [{
        type: "retrieval", sources: sources, refused: false, min_score: MIN_SCORE,
        chunk_count: chunks.length, corpus_count: (d.parallel || []).length,
        embedding_mode: "static", embedding_model: "本地检索（字符二元组余弦）",
        llm_mode: "static", llm_model: "抽取式作答"
      }];
      var step = 14;
      for (var i = 0; i < answer.length; i += step) {
        events.push({ type: "delta", content: answer.slice(i, i + step) });
      }
      events.push({ type: "done", mode: "static", elapsed_ms: 260 + answer.length * 3 });
      return sseResponse(events);
    });
  }

  function handleTranslate(body) {
    return loadAll().then(function (d) {
      loadAllCache = d;
      var started = Date.now();
      var text = String(body.text || "");
      var sourceLang = normalizeLang(body.source);
      var targetLang = normalizeLang(body.target);
      var terms = matchTerms(text, sourceLang, targetLang);
      var refs = findReferences(d, text, sourceLang, targetLang);
      var news = findNews(d, text);
      var translation = sourceLang === targetLang ? text : translateText(text, sourceLang, targetLang, terms, refs);
      return {
        ok: true,
        translation: translation,
        matches: terms.map(function (h) {
          return {
            source_term: h.source_term, target_term: h.target_term,
            domain: h.domain, review_status: h.review_status,
            position: h.position, length: h.length
          };
        }),
        references: refs.map(function (r) {
          return {
            source_text: r.source_text, target_text: r.target_text,
            lang_pair: r.lang_pair, domain: r.domain, score: r.score, source: r.source
          };
        }),
        news_sources: news,
        elapsed_ms: Date.now() - started,
        mode: "static",
        model: "本地术语 + 语料检索",
        source_lang: sourceLang,
        target_lang: targetLang
      };
    });
  }

  /* ---------------- fetch 拦截 ---------------- */
  var rawFetch = window.fetch ? window.fetch.bind(window) : null;

  window.fetch = function (input, init) {
    var url = typeof input === "string" ? input : (input && input.url) || "";
    var path = url.split("?")[0];
    if (path !== "/api/status" && path !== "/api/chat" && path !== "/api/translate" && path !== "/api/rebuild") {
      return rawFetch ? rawFetch(input, init) : Promise.reject(new Error("fetch unavailable"));
    }
    var method = ((init && init.method) || (input && input.method) || "GET").toUpperCase();
    var body = {};
    if (init && init.body) {
      try { body = JSON.parse(init.body); } catch (e) { body = {}; }
    }
    if (path === "/api/rebuild") {
      return jsonResponse({ ok: true, message: "静态部署：索引在浏览器内实时构建，无需重建。" });
    }
    if (path === "/api/status") {
      return loadAll().then(statusPayload).then(jsonResponse);
    }
    if (method !== "POST") return jsonResponse({ ok: false, error: "仅支持 POST" });
    if (path === "/api/chat") return handleChat(body);
    if (path === "/api/translate") return handleTranslate(body).then(jsonResponse);
    return jsonResponse({ ok: false, error: "未知接口" });
  };
})();
