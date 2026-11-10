/*
 * 云端后端地址配置
 * ------------------------------------------------------------
 * 填入 Cloudflare Worker 的地址后（例如 https://asean-kb.xxx.workers.dev），
 * 问答与翻译将调用 BigModel 向量检索 + DeepSeek 大模型。
 * 留空则自动使用浏览器内的本地检索（抽取式作答），页面不会白屏。
 */
window.KB_API_BASE = "";

/* 是否已配置云端后端（供页面显示状态） */
window.KB_API_ON = !!window.KB_API_BASE;
