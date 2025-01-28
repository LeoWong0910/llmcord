<h1 align="center">
  llmcord
</h1>

<h3 align="center"><i>
  Talk to LLMs with your friends!
</i></h3>

<p align="center">
  <img src="https://github.com/jakobdylanc/llmcord/assets/38699060/789d49fe-ef5c-470e-b60e-48ac03057443" alt="">
</p>

llmcord transforms Discord into a collaborative LLM frontend. It works with practically any LLM, remote or locally hosted.

## Features

### Reply-based chat system
Just @ the bot to start a conversation and reply to continue. Build conversations with reply chains!

You can:
- Branch conversations endlessly
- Continue other people's conversations
- @ the bot while replying to ANY message to include it in the conversation

Additionally:
- When DMing the bot, conversations continue automatically (no reply required). To start a fresh conversation, just @ the bot. You can still reply to continue from anywhere.
- You can branch conversations into [threads](https://support.discord.com/hc/en-us/articles/4403205878423-Threads-FAQ). Just create a thread from any message and @ the bot inside to continue.
- Back-to-back messages from the same user are automatically chained together. Just reply to the latest one and the bot will see all of them.

### Choose any LLM
llmcord supports remote models from:
- [OpenAI API](https://platform.openai.com/docs/models)
- [xAI API](https://docs.x.ai/docs/models)
- [Mistral API](https://docs.mistral.ai/getting-started/models/models_overview)
- [Groq API](https://console.groq.com/docs/models)
- [OpenRouter API](https://openrouter.ai/models)

Or run a local model with:
- [Ollama](https://ollama.com)
- [LM Studio](https://lmstudio.ai)
- [vLLM](https://github.com/vllm-project/vllm)

...Or use any other OpenAI compatible API server.

### And more:
- Supports image attachments when using a vision model (like gpt-4o, claude-3, llava, etc.)
- Supports text file attachments (.txt, .py, .c, etc.)
- Customizable personality (aka system prompt)
- User identity aware (OpenAI API and xAI API only)
- Streamed responses (turns green when complete, automatically splits into separate messages when too long)
- Hot reloading config (you can change settings without restarting the bot)
- Displays helpful warnings when appropriate (like "⚠️ Only using last 25 messages" when the customizable message limit is exceeded)
- Caches message data in a size-managed (no memory leaks) and mutex-protected (no race conditions) global dictionary to maximize efficiency and minimize Discord API calls
- Fully asynchronous
- 1 Python file, ~200 lines of code

## Instructions

1. Clone the repo:
   ```bash
   git clone https://github.com/jakobdylanc/llmcord
   ```

2. Create a copy of "config-example.yaml" named "config.yaml" and set it up:

### Discord settings:

| Setting | Description |
| --- | --- |
| **bot_token** | Create a new Discord bot at [discord.com/developers/applications](https://discord.com/developers/applications) and generate a token under the "Bot" tab. Also enable "MESSAGE CONTENT INTENT". |
| **client_id** | Found under the "OAuth2" tab of the Discord bot you just made. |
| **status_message** | Set a custom message that displays on the bot's Discord profile. **Max 128 characters.** |
| **max_text** | The maximum amount of text allowed in a single message, including text from file attachments.<br />(Default: `100,000`) |
| **max_images** | The maximum number of image attachments allowed in a single message. **Only applicable when using a vision model.**<br />(Default: `5`) |
| **max_messages** | The maximum number of messages allowed in a reply chain. When exceeded, the oldest messages are dropped.<br />(Default: `25`) |
| **use_plain_responses** | When set to `true` the bot will use plaintext responses instead of embeds. Plaintext responses have a shorter character limit so the bot's messages may split more often. **Also disables streamed responses and warning messages.**<br />(Default: `false`) |
| **allow_dms** | Set to `false` to disable direct message access.<br />(Default: `true`) |
| **permissions** | Configure permissions for `users`, `roles` and `channels`, each with a list of `allowed_ids` and `blocked_ids`. **Leave `allowed_ids` empty to allow ALL. Role and channel permissions do not affect DMs. You can control channel permissions in groups using [category](https://support.discord.com/hc/en-us/articles/115001580171-Channel-Categories-101) IDs.** |

### LLM settings:

| Setting | Description |
| --- | --- |
| **providers** | Add the LLM providers you want to use, each with a `base_url` and optional `api_key` entry. Popular providers (`openai`, `ollama`, etc.) are already included. **Only supports OpenAI compatible APIs.** |
| **model** | Set to `<provider name>/<model name>`, e.g:<br /><br />-`openai/gpt-4o`<br />-`ollama/llama3.3`<br />-`openrouter/anthropic/claude-3.5-sonnet` |
| **extra_api_parameters** | Extra API parameters for your LLM. Add more entries as needed. **Refer to your provider's documentation for supported API parameters.**<br />(Default: `max_tokens=4096, temperature=1.0`) |
| **system_prompt** | Write anything you want to customize the bot's behavior! **Leave blank for no system prompt.** |

3. Run the bot:

   **No Docker:**
   ```bash
   python -m pip install -U -r requirements.txt
   python llmcord.py
   ```

   **With Docker:**
   ```bash
   docker compose up
   ```

## Notes

- If you're having issues, try my suggestions [here](https://github.com/jakobdylanc/llmcord/issues/19)

- Only models from OpenAI API and xAI API are "user identity aware" because only they support the "name" parameter in the message object. Hopefully more providers support this in the future.

- PRs are welcome :)

## Star History

<a href="https://star-history.com/#jakobdylanc/llmcord&Date">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=jakobdylanc/llmcord&type=Date&theme=dark" />
    <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=jakobdylanc/llmcord&type=Date" />
    <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=jakobdylanc/llmcord&type=Date" />
  </picture>
</a>

# Discord AI Assistant

基於 [llmcord](https://github.com/jakobdylanc/llmcord) 的客製化 Discord AI 助手，支持圖片分析和多輪對話。

## 快速開始

1. 配置設置：
```bash
# 複製示例配置文件
cp config-example.yaml config.yaml

# 編輯配置文件，填入你的設置
nano config.yaml
```

2. 使用 Docker 部署：
```bash
# 構建並啟動
docker compose up -d

# 查看日誌
docker compose logs -f
```

3. 停止服務：
```bash
docker compose down
```

## 主要功能

- 支持多種 LLM 模型
- 支持圖片分析
- 支持中英文對話
- 支持多輪對話
- 支持文本文件處理
- 支持代幣使用統計
- 支持成本追蹤

## 常用命令

基礎命令：
- `/chat [消息]` - 與AI助手對話
- `/help` - 顯示幫助信息
- `/tokens` - 查看令牌使用統計
- `/cost` - 查看成本統計
- `/daily` - 查看每日統計
- `/info` - 查看機器人信息

管理命令（需要管理員權限）：
- `/sync` - 同步斜杠命令
- `/reset` - 重置統計數據
- `/config` - 查看當前配置

## 注意事項

1. 請確保 config.yaml 中的 token 和 API key 已正確設置
2. 數據文件保存在 data 目錄中
3. 日誌可通過 docker compose logs 查看
4. 支持熱重載配置（無需重啟即可更改設置）
5. 自動分割過長消息
6. 支持流式輸出

## 環境要求

- Docker
- Docker Compose

## 技術支持

如有問題，請提交 Issue 或加入我們的 Discord 社群。

## 授權協議

本項目基於 MIT 協議開源。
