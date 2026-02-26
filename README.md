# ComfyUI-ClipAPI-ZaKo

随机提示词固定人物

<img width="800" alt="workflow示例" src="https://github.com/user-attachments/assets/f18cb60c-efe2-478d-b6e9-1c8c1d17a5db" />

## 功能特性

- **固定人物提示词**：人物提示词为固定内容，LLM 默认不对其进行更改
- **随机提示词兼容**：支持 WeiLin 节点（只需最终输出为 text 格式即可）
- **智能冲突处理**：LLM 接收人物提示词与随机提示词，若随机提示词中出现冲突内容则自动删除，以达到固定人物的效果（画师串同理）
- **自定义元提示词**：可自定义 LLM 的元提示词，以适配不同场景优化需求

## 注意事项

⚠️ **重要提示**：
- 本项目仅适配硅基流动 API，若需其他 API 建议自行调整
- 填入密钥时分享工作流会被一同分享，容易造成密钥泄露，**强烈建议**搭配本人的另一个插件一起使用

## 相关项目

- [ComfyUI-MetaData-ZaKo](https://github.com/xiaoshengyvlin/ComfyUI-MetaData-ZaKo) - 图片元信息置换插件

## 免责声明

本人小白，代码纯 AI 生成，无手工修改，无能力更新及维护
