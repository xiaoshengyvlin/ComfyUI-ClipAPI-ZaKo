import threading
import logging
from typing import Dict, Optional, Tuple, Any

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# 日志配置
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - ZaKo提示词融合器 - %(levelname)s - %(message)s"
)
logger = logging.getLogger("ZaKoPromptMerger")


class ZaKoPromptMerger:
    API_URL = "https://api.siliconflow.cn/v1/chat/completions"

    # 跑图默认优化参数
    DEFAULT_CONNECT_TIMEOUT = 10
    DEFAULT_READ_TIMEOUT = 60
    DEFAULT_MAX_TOKENS = 1500
    DEFAULT_TEMPERATURE = 0.1
    DEFAULT_RETRY_TOTAL = 3

    # 线程本地Session：串行跑图复用连接，提升速度
    _session_local = threading.local()

    # 跑图计数：日志标记，方便定位问题
    _run_count = 0
    _count_lock = threading.Lock()

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        DEFAULT_PROMPT = """# AI绘画提示词融合专家（ZaKo优化版）

你是一个精密、严谨的AI绘画提示词处理引擎。你的唯一职责是根据以下**不可违反的规则**，对输入的内容进行融合与格式化，输出可直接用于AI绘画工具（如Stable Diffusion）的提示词。

## 🔧 核心处理规则 (The Law)

### 1. 基准原则：人物提示词神圣不可侵犯
*   **锁定**：将用户提供的【人物提示词】部分视为**绝对基准**。在整个处理过程中，**不得对其内容进行任何形式的修改、删减、添加或重新排序**。

### 2. 权重优先原则：保护所有语法标签
*   **完整保留**：任何带有显式权重语法（如 `(artist:miv4t:1.10)`、`(tag:1.2)`、`[tag:0.9]`) 的标签，必须**原封不动地保留其完整形式**，包括括号、冒号、权重数值。
*   **画师标签**：无显式权重的画师/风格标签（如 `artist:ratatatat74`, `style of greg rutkowski`）也保持原样。

### 3. 冲突消除原则：严格的定义与执行
*   **冲突定义（唯一标准）**：**仅当【随机提示词】中的某个标签，试图直接修改或替换【人物提示词】中已经明确设定的、具体的、可视觉化的特征时，才构成冲突。**
    *   **✅ 冲突示例（必须删除）**：人物提示词为 `green hair`，随机提示词中出现 `blue hair` 或 `red hair`。
    *   **❌ 非冲突示例（必须保留）**：
        *   人物为 `1girl`，随机提示词中出现 `1boy`，且两者之间存在互动则视为场景中的交互角色，包括但不限于性暗示互动，若仅出现男性及描述，无互动姿势则删去男性及其描述。
        *   任何关于动作（`running`）、姿态（`sitting`）、场景（`in a forest`）、氛围（`moody lighting`）或NSFW内容的描述，只要不改变人物自身的**既定核心特征**（如发型、发色、瞳色、服饰）则视为非冲突。
*   **处理逻辑**：对【随机提示词】进行逐项检查。**仅删除明确符合上述"冲突定义"的标签**。对于所有其他标签，包括你不确定是否构成冲突的，**一律默认保留**。

### 4. 风格聚合原则：合并画师与风格
*   **来源**：从【画师串】和【随机提示词】中，提取所有画师名（`by ...`, `artist:...`, `style of ...`）及风格标签（如 `detailed background`, `anime screencap`）。
*   **操作**：将提取到的所有画师与风格标签**合并为一个列表，并进行去重**，形成最终的**总画师风格串**。

### 5. 多角色结构保留原则（若无，则忽略）
*   **识别模式**：识别并保留原始输入中的多角色结构（如 `char1：...`, `char2：...`, `char3：...`）。
*   **处理方式**：将多角色描述作为独立的段落进行处理，不与其他内容合并。

### 6. 主动增强原则：智能补充质量标签
*   **质量增强**：根据画面内容智能添加通用的高质量标签，如：
    *   **画质类**：`best quality`, `masterpiece`, `high resolution`, `ultra detailed`, `sharp focus`
    *   **光照类**：`masterpiece lighting`, `cinematic lighting`, `dramatic lighting`
    *   **细节类**：`intricate details`, `fine details`, `detailed background`
*   **智能判断**：根据现有内容判断是否需要补充：
    *   如果已有类似标签，则不重复添加
    *   优先添加到【总画师风格串】的开头部分
    *   避免与现有内容冲突

### 7. 输出格式化原则：多段式固定结构
最终输出必须严格遵循以下格式，每段以英文逗号结尾并换行：
[主动增强词],
[总画师风格串],
[人物提示词],
[场景描述],
[角色1描述],
[角色2描述],
[角色3描述],
[...其他角色描述]"""

        return {
            "optional": {
                # 提示词输入区
                "人物提示词": ("STRING", {"forceInput": True}),
                "随机提示词": ("STRING", {"forceInput": True}),
                "画师串": ("STRING", {"forceInput": True}),
                "备用1": ("STRING", {"forceInput": True}),
                "备用2": ("STRING", {"forceInput": True}),

                # 【密钥输入区】恢复纯手动输入，填了就用
                "硅基流动密钥": ("STRING", {
                    "default": "",
                    "label": "硅基流动密钥（必填，分享工作流前请清空！）",
                    "placeholder": "sk-xxx 从硅基流动控制台获取",
                    "widget": "textbox"
                }),

                # 模型配置（默认DeepSeek-V3.2）
                "模型名称": ("STRING", {
                    "default": "deepseek-ai/DeepSeek-V3.2",
                    "label": "模型名称",
                    "placeholder": "如：deepseek-ai/DeepSeek-V3.2",
                    "widget": "textbox"
                }),
                "提示词融合指令": ("STRING", {
                    "default": DEFAULT_PROMPT,
                    "label": "提示词融合规则指令",
                    "multiline": True,
                    "widget": "textbox"
                }),

                # API调用参数（已隐藏SSL和随机种子）
                "温度": ("FLOAT", {
                    "default": cls.DEFAULT_TEMPERATURE,
                    "min": 0.0, "max": 2.0, "step": 0.05
                }),
                "最大输出Token": ("INT", {
                    "default": cls.DEFAULT_MAX_TOKENS,
                    "min": 64, "max": 8192, "step": 64
                }),
                "连接超时秒": ("INT", {
                    "default": cls.DEFAULT_CONNECT_TIMEOUT,
                    "min": 2, "max": 120, "step": 1
                }),
                "读取超时秒": ("INT", {
                    "default": cls.DEFAULT_READ_TIMEOUT,
                    "min": 5, "max": 300, "step": 1
                }),
                "失败重试次数": ("INT", {
                    "default": cls.DEFAULT_RETRY_TOTAL,
                    "min": 0, "max": 10, "step": 1,
                    "label": "同模型失败重试次数"
                }),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("融合后提示词",)
    FUNCTION = "merge_prompts"
    CATEGORY = "ZaKo"
    DESCRIPTION = "ZaKo提示词融合器（纯手动密钥·默认DeepSeek-V3.2·无缓存·跑图专用）"

    # ---------- 极简工具函数 ----------
    @staticmethod
    def _trim(s: Optional[str]) -> str:
        return (s or "").strip()

    @staticmethod
    def _clamp_num(v: int | float, min_val: int | float, max_val: int | float) -> int | float:
        return max(min_val, min(max_val, v))

    @classmethod
    def _add_run_count(cls) -> int:
        with cls._count_lock:
            cls._run_count += 1
            return cls._run_count

    @classmethod
    def _get_session(cls, retry_times: int) -> requests.Session:
        retry_times = int(cls._clamp_num(retry_times, 0, 10))
        if not hasattr(cls._session_local, "sessions"):
            cls._session_local.sessions = {}
        if retry_times in cls._session_local.sessions:
            return cls._session_local.sessions[retry_times]

        # 配置Session重试和连接池
        sess = requests.Session()
        retry_config = Retry(
            total=retry_times,
            connect=retry_times,
            read=retry_times,
            backoff_factor=0.5,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=frozenset(["POST"]),
            raise_on_status=False
        )
        adapter = HTTPAdapter(max_retries=retry_config, pool_connections=20, pool_maxsize=50)
        sess.mount("https://", adapter)
        sess.mount("http://", adapter)
        cls._session_local.sessions[retry_times] = sess
        return sess

    @staticmethod
    def _get_error_detail(resp: requests.Response) -> str:
        try:
            data = resp.json()
            if isinstance(data, dict):
                if isinstance(data.get("error"), dict):
                    return str(data["error"].get("message", "")).strip()
                if "message" in data:
                    return str(data.get("message", "")).strip()
        except Exception:
            pass
        txt = (resp.text or "").strip()
        return txt[:200] if txt else ""

    @staticmethod
    def _parse_api_result(data: Dict[str, Any]) -> str:
        content = (
            data.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
        )
        return content.strip() if isinstance(content, str) else ""

    def _call_api(
        self,
        session: requests.Session,
        api_key: str,
        model_name: str,
        final_prompt: str,
        temperature: float,
        max_tokens: int,
        connect_timeout: int,
        read_timeout: int,
        seed: int,
        verify_ssl: bool
    ) -> Tuple[Optional[str], int, str]:
        headers = {
            "Content-Type": "application/json; charset=utf-8",
            "Authorization": f"Bearer {api_key}"
        }
        payload = {
            "model": model_name,
            "messages": [{"role": "user", "content": final_prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False
        }
        if seed >= 0:
            payload["seed"] = seed

        resp = session.post(
            self.API_URL,
            json=payload,
            headers=headers,
            timeout=(connect_timeout, read_timeout),
            verify=verify_ssl
        )

        if resp.status_code >= 400:
            error_detail = self._get_error_detail(resp)
            return None, resp.status_code, error_detail

        try:
            result_data = resp.json()
        except ValueError:
            return None, resp.status_code, "API返回了非JSON格式内容"

        final_result = self._parse_api_result(result_data)
        if not final_result:
            return None, 200, "API返回了空内容"
        return final_result, 200, ""

    def merge_prompts(self, **kwargs: Any) -> Tuple[str]:
        try:
            current_run_count = self._add_run_count()
            logger.info(f"===== 开始处理第{current_run_count}张图的随机提示词 =====")

            # 【核心】直接读取手动输入的密钥，无任何复杂逻辑
            api_key = self._trim(kwargs.get("硅基流动密钥", ""))
            if not api_key:
                error_msg = f"❌ 第{current_run_count}张图失败：请填写硅基流动密钥"
                logger.error(error_msg)
                return (error_msg,)
            if not api_key.startswith("sk-"):
                error_msg = f"❌ 第{current_run_count}张图失败：密钥格式错误，应为sk-xxx开头"
                logger.error(error_msg)
                return (error_msg,)

            # 读取其他参数
            model_name = self._trim(kwargs.get("模型名称", "deepseek-ai/DeepSeek-V3.2")) or "deepseek-ai/DeepSeek-V3.2"
            prompt_rule = self._trim(
                kwargs.get("提示词融合指令", self.INPUT_TYPES()["optional"]["提示词融合指令"][1]["default"])
            )

            temperature = float(self._clamp_num(float(kwargs.get("温度", self.DEFAULT_TEMPERATURE)), 0.0, 2.0))
            max_tokens = int(self._clamp_num(int(kwargs.get("最大输出Token", self.DEFAULT_MAX_TOKENS)), 64, 8192))
            connect_timeout = int(self._clamp_num(int(kwargs.get("连接超时秒", self.DEFAULT_CONNECT_TIMEOUT)), 2, 120))
            read_timeout = int(self._clamp_num(int(kwargs.get("读取超时秒", self.DEFAULT_READ_TIMEOUT)), 5, 300))
            retry_times = int(self._clamp_num(int(kwargs.get("失败重试次数", self.DEFAULT_RETRY_TOTAL)), 0, 10))
            
            # 隐藏参数固定默认值
            verify_ssl = True  # 强制开启SSL安全校验
            seed = -1  # 不设置随机种子

            # 收集提示词
            prompt_order = ["人物提示词", "随机提示词", "画师串", "备用1", "备用2"]
            seen_prompt = set()
            prompt_items = []
            for prompt_name in prompt_order:
                prompt_content = self._trim(kwargs.get(prompt_name, ""))
                if not prompt_content:
                    continue
                prompt_key = f"{prompt_name}:{prompt_content}"
                if prompt_key in seen_prompt:
                    continue
                seen_prompt.add(prompt_key)
                prompt_items.append((prompt_name, prompt_content))

            if not prompt_items:
                error_msg = f"❌ 第{current_run_count}张图失败：至少输入1个有效提示词"
                logger.error(error_msg)
                return (error_msg,)

            # 拼接最终发给API的提示词
            prompt_list_text = "\n".join([f"{i+1}. 【{name}】{content}" for i, (name, content) in enumerate(prompt_items)])
            final_api_prompt = f"{prompt_rule}\n\n待融合提示词：\n{prompt_list_text}"

            # 实时调用API，无缓存
            session = self._get_session(retry_times)
            try:
                result, status_code, error_detail = self._call_api(
                    session=session,
                    api_key=api_key,
                    model_name=model_name,
                    final_prompt=final_api_prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    connect_timeout=connect_timeout,
                    read_timeout=read_timeout,
                    seed=seed,
                    verify_ssl=verify_ssl
                )
                if result:
                    logger.info(f"第{current_run_count}张图：模型【{model_name}】调用成功，已返回融合结果")
                    return (result,)
                else:
                    final_error = f"❌ 第{current_run_count}张图失败：模型【{model_name}】调用失败，HTTP {status_code} - {error_detail}"
                    logger.error(final_error)
                    return (final_error,)
            except requests.exceptions.Timeout:
                final_error = f"❌ 第{current_run_count}张图失败：模型【{model_name}】请求超时"
                logger.error(final_error)
                return (final_error,)
            except requests.exceptions.SSLError:
                final_error = f"❌ 第{current_run_count}张图失败：模型【{model_name}】SSL校验失败"
                logger.error(final_error)
                return (final_error,)
            except requests.exceptions.RequestException as e:
                final_error = f"❌ 第{current_run_count}张图失败：模型【{model_name}】网络异常-{type(e).__name__}"
                logger.error(final_error)
                return (final_error,)
            except Exception as e:
                final_error = f"❌ 第{current_run_count}张图失败：模型【{model_name}】未知错误-{str(e)[:120]}"
                logger.error(final_error, exc_info=True)
                return (final_error,)

        except Exception as e:
            final_error = f"❌ 第{current_run_count}张图失败：{str(e)}"
            logger.error(final_error, exc_info=True)
            return (final_error,)


# ComfyUI节点注册
NODE_CLASS_MAPPINGS = {"ZaKoPromptMerger": ZaKoPromptMerger}
NODE_DISPLAY_NAME_MAPPINGS = {"ZaKoPromptMerger": "ZaKo提示词融合器"}
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]