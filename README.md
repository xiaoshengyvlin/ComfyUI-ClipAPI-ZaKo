# ComfyUI-ClipAPI-ZaKo

随机提示词固定人物

<img width="800" alt="workflow示例" src="https://github.com/user-attachments/assets/f18cb60c-efe2-478d-b6e9-1c8c1d17a5db" />

## 功能特性

- **固定人物提示词**：人物提示词为固定内容，LLM 默认不对其进行更改
- **随机提示词兼容**：支持 WeiLin 节点（只需最终输出为 text 格式即可）
- **智能冲突处理**：LLM 接收人物提示词与随机提示词，若随机提示词中出现冲突内容则自动删除，以达到固定人物的效果（画师串同理）
- **自定义元提示词**：可自定义 LLM 的元提示词，以适配不同场景优化需求
- **多 API 适配**：预设硅基流动 / OpenAI / DeepSeek / 通义千问 / 智谱GLM / 月之暗面Kimi / Ollama本地，也可手动填写任意 OpenAI 兼容地址
- **密钥本地存储**：API 密钥仅存于本机浏览器，不随工作流序列化，分享工作流不会泄露密钥

## 注意事项

⚠️ **重要提示**：
- 支持任意 OpenAI 兼容 API，选择「API提供商」后自动填入对应地址，也可手动填写自定义地址
- API 密钥仅存本机浏览器 localStorage，按提供商分别保存，分享工作流时不会包含密钥

## 相关项目

- [ComfyUI-MetaData-ZaKo](https://github.com/xiaoshengyvlin/ComfyUI-MetaData-ZaKo) - 图片元信息置换插件
