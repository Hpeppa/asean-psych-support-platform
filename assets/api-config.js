/*
 * 云端后端地址配置
 * ------------------------------------------------------------
 * 指向 Netlify Functions 后端（BigModel 向量检索 + DeepSeek 流式回答）。
 * 留空则自动使用浏览器内的本地检索（抽取式作答），页面不会白屏。
 */
window.KB_API_BASE = "https://rococo-raindrop-88656d.netlify.app";

/* 是否已配置云端后端（供页面显示状态） */
window.KB_API_ON = !!window.KB_API_BASE;