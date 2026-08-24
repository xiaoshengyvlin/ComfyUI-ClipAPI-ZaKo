import { app } from "../../../scripts/app.js";


// 默认地址映射需与 Python 端 PROVIDERS 保持一致
const PROVIDER_BASES = {
    "硅基流动": "https://api.siliconflow.cn/v1",
    "OpenAI": "https://api.openai.com/v1",
    "DeepSeek": "https://api.deepseek.com/v1",
    "通义千问": "https://dashscope.aliyuncs.com/compatible-mode/v1",
    "智谱GLM": "https://open.bigmodel.cn/api/paas/v4",
    "月之暗面Kimi": "https://api.moonshot.cn/v1",
    "Ollama本地": "http://localhost:11434/v1",
};

const KEY_STORAGE_PREFIX = "zako_clip_api_key";


app.registerExtension({
    name: "ZaKoPromptMerger.ApiKey",

    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "ZaKoPromptMerger") return;

        // 提供商列表从节点定义动态读取，与 Python 端单一来源
        const definedOptions = nodeData.input?.optional?.["API提供商"];
        const providerOptions = Array.isArray(definedOptions) && definedOptions.length
            ? definedOptions
            : Object.keys(PROVIDER_BASES);

        const origOnNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const r = origOnNodeCreated?.apply(this, arguments);

            const apiKeyWidget = this.widgets.find((w) => w.name === "硅基流动密钥");
            const providerWidget = this.widgets.find((w) => w.name === "API提供商");
            const apiBaseWidget = this.widgets.find((w) => w.name === "API地址");
            if (!apiKeyWidget || !providerWidget || !apiBaseWidget) return r;

            const loadKey = (provider) => localStorage.getItem(`${KEY_STORAGE_PREFIX}_${provider}`) || "";
            const saveKey = (provider, value) => localStorage.setItem(`${KEY_STORAGE_PREFIX}_${provider}`, value || "");

            let curProvider = providerWidget.value || providerOptions[0];

            // 密钥不随工作流序列化，仅存本机浏览器
            apiKeyWidget.serialize = false;
            apiKeyWidget.value = loadKey(curProvider);
            const origKeyCallback = apiKeyWidget.callback;
            apiKeyWidget.callback = function (value) {
                saveKey(curProvider, value);
                return origKeyCallback?.call(this, value);
            };

            // 切换提供商：保存当前密钥、加载新密钥、自动填入对应地址
            const origProviderCallback = providerWidget.callback;
            providerWidget.callback = function (value) {
                saveKey(curProvider, apiKeyWidget.value);
                curProvider = value;
                apiKeyWidget.value = loadKey(curProvider);
                const base = PROVIDER_BASES[value];
                if (base) apiBaseWidget.value = base;
                return origProviderCallback?.call(this, value);
            };

            return r;
        };
    },
});
