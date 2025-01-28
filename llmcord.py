import asyncio
from base64 import b64encode
from dataclasses import dataclass, field
from datetime import datetime as dt
import logging
from typing import Literal, Optional
import json
from datetime import datetime, timedelta

import discord
import httpx
from openai import AsyncOpenAI
import yaml
from discord import app_commands
from discord.ext import commands

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
)

VISION_MODEL_TAGS = (
    "gpt-4o", 
    "claude-3", 
    "claude-3-opus",
    "claude-3-sonnet", 
    "claude-2", 
    "deepseek-vision",
    "deepseek-vl",
    "gemini", 
    "pixtral", 
    "llava", 
    "vision", 
    "vl"
)

PROVIDERS_SUPPORTING_USERNAMES = (
    "openai", 
    "x-ai",
    "claude",
    "deepseek"
)

ALLOWED_FILE_TYPES = ("image", "text")

EMBED_COLOR_COMPLETE = discord.Color.dark_green()
EMBED_COLOR_INCOMPLETE = discord.Color.orange()

STREAMING_INDICATOR = " ⚪"
EDIT_DELAY_SECONDS = 1

MAX_MESSAGE_NODES = 100

# Add a global variable to track token usage
token_usage = {
    'total_tokens': 0,
    'completion_tokens': 0,
    'prompt_tokens': 0,
    'last_reset': datetime.now().isoformat(),
    'initial_balance_usd': 100.00,
    'initial_balance_rmb': 720.00,
    'total_cost': 0.00
}

# Add function to save/load token usage
def save_token_usage():
    with open('token_usage.json', 'w') as f:
        json.dump(token_usage, f)

def load_token_usage():
    try:
        with open('token_usage.json', 'r') as f:
            global token_usage
            token_usage = json.load(f)
    except FileNotFoundError:
        save_token_usage()


def get_config(filename="config.yaml"):
    with open(filename, "r") as file:
        return yaml.safe_load(file)


cfg = get_config()

if client_id := cfg["client_id"]:
    invite_permissions = discord.Permissions(
        # 基本权限
        send_messages=True,          # 发送消息
        view_channel=True,           # 查看频道（原 read_messages）
        read_message_history=True,   # 读取消息历史
        embed_links=True,            # 嵌入链接
        attach_files=True,           # 附加文件
        use_external_emojis=True,    # 使用外部表情
        add_reactions=True,          # 添加反应
        # 不需要 application_commands，因为它是在 scope 中设置的
    )
    
    # 在 scope 中添加 applications.commands
    invite_url = f"https://discord.com/api/oauth2/authorize?client_id={client_id}&permissions={invite_permissions.value}&scope=bot%20applications.commands"
    logging.info(f"\n\nBOT INVITE URL:\n{invite_url}\n")

intents = discord.Intents.default()
intents.message_content = True
activity = discord.CustomActivity(name=(cfg["status_message"] or "github.com/jakobdylanc/llmcord")[:128])

class LLMClient(discord.Client):
    def __init__(self):
        super().__init__(intents=intents, activity=activity)
        self.tree = app_commands.CommandTree(self)
        # 添加一个标志来追踪是否已同步命令
        self.synced = False

    async def setup_hook(self):
        # 只在第一次启动时同步命令
        if not self.synced:
            await self.tree.sync()
            self.synced = True
            logging.info("Slash commands synced!")

discord_client = LLMClient()

httpx_client = httpx.AsyncClient()

msg_nodes = {}
last_task_time = 0

# 添加模型价格信息（每1000个tokens的价格，单位：美元）
MODEL_PRICES = {
    'claude/claude-3-5-sonnet-20241022': {
        'prompt': 0.03,
        'completion': 0.06
    },
    'claude/claude-2': {
        'prompt': 0.02,
        'completion': 0.04
    },
    'deepseek/deepseek-vision': {
        'prompt': 0.02,
        'completion': 0.04
    },
    'deepseek/deepseek-vl': {
        'prompt': 0.02,
        'completion': 0.04
    }
}

@dataclass
class MsgNode:
    text: Optional[str] = None
    images: list = field(default_factory=list)

    role: Literal["user", "assistant"] = "assistant"
    user_id: Optional[int] = None

    next_msg: Optional[discord.Message] = None

    has_bad_attachments: bool = False
    fetch_next_failed: bool = False

    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


@discord_client.event
async def on_message(new_msg):
    if new_msg.author.bot:
        return

    # 检查是否需要处理消息
    is_dm = new_msg.channel.type == discord.ChannelType.private
    if not is_dm and discord_client.user not in new_msg.mentions:
        return

    # 检查权限
    role_ids = tuple(role.id for role in getattr(new_msg.author, "roles", ()))
    channel_ids = tuple(id for id in (new_msg.channel.id, getattr(new_msg.channel, "parent_id", None), getattr(new_msg.channel, "category_id", None)) if id)

    cfg = get_config()
    allow_dms = cfg["allow_dms"]
    
    # 检查权限设置
    if isinstance(cfg["permissions"], dict):
        permissions = cfg["permissions"]
        (allowed_user_ids, blocked_user_ids), (allowed_role_ids, blocked_role_ids), (allowed_channel_ids, blocked_channel_ids) = (
            (perm["allowed_ids"], perm["blocked_ids"]) for perm in (permissions["users"], permissions["roles"], permissions["channels"])
        )
    else:
        allowed_user_ids = blocked_user_ids = allowed_role_ids = blocked_role_ids = allowed_channel_ids = blocked_channel_ids = []

    allow_all_users = not allowed_user_ids if is_dm else not allowed_user_ids and not allowed_role_ids
    is_good_user = allow_all_users or new_msg.author.id in allowed_user_ids or any(id in allowed_role_ids for id in role_ids)

    if not is_good_user:
        return

    # 处理 @ 消息
    if discord_client.user in new_msg.mentions:
        try:
            # 获取聊天历史
            cfg = get_config()
            max_messages = cfg["max_messages"]
            history = await get_chat_history(new_msg.channel, new_msg, max_messages - 1)
            
            # 获取最近的连续消息
            messages = []
            current_author = new_msg.author
            async for msg in new_msg.channel.history(limit=20, before=new_msg):
                if msg.author != current_author:
                    break
                if discord_client.user not in msg.mentions:
                    continue
                content = msg.content.replace(discord_client.user.mention, '').strip()
                if content:
                    messages.insert(0, content)
            
            # 添加当前消息
            current_content = new_msg.content.replace(discord_client.user.mention, '').strip()
            if current_content:
                messages.append(current_content)
            
            # 如果没有有效消息，直接返回
            if not messages:
                return
                
            # 合并连续消息
            combined_message = "\n".join(messages)
            
            # 添加当前消息到历史记录
            history.append({
                "role": "user",
                "content": combined_message
            })
            
            # 添加系统提示
            if system_prompt := cfg["system_prompt"]:
                history.append({
                    "role": "system",
                    "content": system_prompt
                })
            
            # 调用 API
            provider, model = cfg["model"].split("/", 1)
            base_url = cfg["providers"][provider]["base_url"]
            api_key = cfg["providers"][provider].get("api_key", "sk-no-key-required")
            openai_client = AsyncOpenAI(base_url=base_url, api_key=api_key)
            
            async with new_msg.channel.typing():
                response_content = ""
                async for chunk in await openai_client.chat.completions.create(
                    model=model,
                    messages=history,
                    stream=True,
                    **cfg["extra_api_parameters"]
                ):
                    if chunk.choices[0].delta.content:
                        response_content += chunk.choices[0].delta.content
                        
                        # 更新 token 使用统计
                        if hasattr(chunk, 'usage') and chunk.usage:
                            update_token_usage(chunk.usage, model)
                
                # 分块发送响应
                if len(response_content) > 2000:
                    chunks = [response_content[i:i+1900] for i in range(0, len(response_content), 1900)]
                    for chunk in chunks:
                        await new_msg.reply(chunk)
                else:
                    await new_msg.reply(response_content)
                    
        except Exception as e:
            logging.exception("Error processing message")
            await new_msg.reply(f"❌ 发生错误：{str(e)}")
            
    return  # 直接返回，不执行后面的代码

async def main():
    load_token_usage()
    await discord_client.start(cfg["bot_token"])


# 修改斜杠命令的实现
@discord_client.tree.command(name="chat", description="与AI助手对话")
async def chat(interaction: discord.Interaction, message: str):
    """与AI助手对话"""
    try:
        await interaction.response.defer()
        
        # 获取聊天历史
        cfg = get_config()
        max_messages = cfg["max_messages"]
        history = await get_chat_history(interaction.channel, interaction.message, max_messages - 1)
        
        # 添加当前消息
        history.append({
            "role": "user",
            "content": message
        })
        
        # 添加系统提示
        if system_prompt := cfg["system_prompt"]:
            history.append({
                "role": "system",
                "content": system_prompt
            })
        
        # 调用 API
        provider, model = cfg["model"].split("/", 1)
        base_url = cfg["providers"][provider]["base_url"]
        api_key = cfg["providers"][provider].get("api_key", "sk-no-key-required")
        openai_client = AsyncOpenAI(base_url=base_url, api_key=api_key)
        
        response_content = ""
        async for chunk in await openai_client.chat.completions.create(
            model=model,
            messages=history,
            stream=True,
            **cfg["extra_api_parameters"]
        ):
            if chunk.choices[0].delta.content:
                response_content += chunk.choices[0].delta.content
                
                # 更新 token 使用统计
                if hasattr(chunk, 'usage') and chunk.usage:
                    update_token_usage(chunk.usage, model)
        
        # 分块发送响应
        if len(response_content) > 2000:
            chunks = [response_content[i:i+1900] for i in range(0, len(response_content), 1900)]
            await interaction.followup.send(chunks[0])
            for chunk in chunks[1:]:
                await interaction.channel.send(chunk)
        else:
            await interaction.followup.send(response_content)
            
    except Exception as e:
        logging.exception("Error in chat command")
        await interaction.followup.send(f"❌ 发生错误：{str(e)}", ephemeral=True)

async def get_chat_history(channel, message, max_messages=10):
    """获取聊天历史"""
    history = []
    try:
        # 获取频道历史消息
        async for msg in channel.history(limit=100, before=message):  # 增加搜索范围
            # 跳过非对话消息
            if msg.author.bot and msg.author != discord_client.user:
                continue
                
            # 构建消息
            content = None
            
            # 处理机器人的回复
            if msg.author == discord_client.user:
                content = msg.content
            
            # 处理 @ 消息
            elif discord_client.user in msg.mentions:
                content = msg.content.replace(discord_client.user.mention, '').strip()
            
            # 处理 /chat 命令消息
            elif msg.interaction and msg.interaction.name == "chat":
                try:
                    content = next((opt.value for opt in msg.interaction.command.options if opt.name == "message"), None)
                except:
                    continue
            
            if content and content.strip():  # 确保内容不为空
                history.append({
                    "role": "assistant" if msg.author == discord_client.user else "user",
                    "content": content.strip()
                })
            
            if len(history) >= max_messages:
                break
                
    except Exception as e:
        logging.error(f"Error getting chat history: {e}")
        
    return list(reversed(history))  # 保持时间顺序

def update_token_usage(usage, model):
    """更新令牌使用统计"""
    try:
        token_usage['total_tokens'] += getattr(usage, 'total_tokens', 0)
        token_usage['completion_tokens'] += getattr(usage, 'completion_tokens', 0)
        token_usage['prompt_tokens'] += getattr(usage, 'prompt_tokens', 0)
        token_usage['conversations'] = token_usage.get('conversations', 0) + 1
        
        # 更新每日统计
        today = datetime.now().strftime('%Y-%m-%d')
        if 'daily' not in token_usage:
            token_usage['daily'] = {}
        token_usage['daily'][today] = token_usage['daily'].get(today, 0) + getattr(usage, 'total_tokens', 0)
        
        # 更新成本
        if model in MODEL_PRICES:
            new_cost = (getattr(usage, 'completion_tokens', 0) / 1000 * MODEL_PRICES[model]['completion'] +
                       getattr(usage, 'prompt_tokens', 0) / 1000 * MODEL_PRICES[model]['prompt'])
            token_usage['total_cost'] = token_usage.get('total_cost', 0) + new_cost
            token_usage['daily'][f"{today}_cost"] = token_usage['daily'].get(f"{today}_cost", 0) + new_cost
        
        save_token_usage()
    except Exception as e:
        logging.error(f"Error updating token usage: {e}")

@discord_client.tree.command(name="tokens", description="查看令牌使用統計")
async def tokens(interaction: discord.Interaction):
    """顯示令牌使用統計"""
    await show_usage_stats(interaction)

@discord_client.tree.command(name="stats", description="查看詳細的使用統計")
async def stats(interaction: discord.Interaction):
    """顯示詳細的使用統計信息"""
    total = token_usage['completion_tokens'] + token_usage['prompt_tokens']
    
    # 计算成本
    cfg = get_config()
    model = cfg["model"]
    cost = 0
    if model in MODEL_PRICES:
        prompt_cost = (token_usage['prompt_tokens'] / 1000) * MODEL_PRICES[model]['prompt']
        completion_cost = (token_usage['completion_tokens'] / 1000) * MODEL_PRICES[model]['completion']
        cost = prompt_cost + completion_cost

    # 获取账户余额信息
    billing = cfg.get('billing', {})
    initial_usd = billing.get('initial_balance_usd', 100.00)
    exchange_rate = billing.get('exchange_rate', 7.20)
    
    # 计算剩余金额
    remaining_usd = initial_usd - token_usage.get('total_cost', 0) - cost
    remaining_rmb = remaining_usd * exchange_rate

    # 计算每日统计
    today = datetime.now().strftime('%Y-%m-%d')
    if 'daily' not in token_usage:
        token_usage['daily'] = {}
    
    daily_tokens = token_usage['daily'].get(today, 0)
    daily_cost = token_usage['daily'].get(f"{today}_cost", 0)

    stats_str = f"""
📊 **使用統計報告** 
開始時間: {token_usage['last_reset']}

💬 **總體令牌使用**
• 總計令牌：{total:,} 個
• 提示令牌：{token_usage['prompt_tokens']:,} 個
• 回覆令牌：{token_usage['completion_tokens']:,} 個
• 對話次數：{token_usage.get('conversations', 0):,} 次

💰 **成本分析**
• 提示成本：${prompt_cost:.4f}
• 回覆成本：${completion_cost:.4f}
• 總計成本：${cost:.4f}

📅 **今日統計**
• 今日令牌：{daily_tokens:,} 個
• 今日成本：${daily_cost:.4f}

💳 **賬戶餘額**
• 剩餘(USD)：${remaining_usd:.2f}
• 剩餘(RMB)：¥{remaining_rmb:.2f}

📈 **效率指標**
• 平均令牌/對話：{total / max(1, token_usage.get('conversations', 1)):,.0f} 個
• 平均成本/千令牌：${(cost * 1000 / max(1, total)):.4f}
"""
    await interaction.response.send_message(stats_str)

@discord_client.tree.command(name="reset", description="重置統計數據（僅管理員）")
async def reset(interaction: discord.Interaction):
    """重置統計數據（僅管理員）"""
    try:
        # 检查是否为服务器所有者或管理员
        if interaction.guild and (interaction.user.id == interaction.guild.owner_id or interaction.user.guild_permissions.administrator):
            token_usage['total_tokens'] = 0
            token_usage['completion_tokens'] = 0
            token_usage['prompt_tokens'] = 0
            token_usage['conversations'] = 0
            token_usage['total_cost'] = 0
            token_usage['last_reset'] = datetime.now().isoformat()
            save_token_usage()
            await interaction.response.send_message("✅ 統計數據已重置！", ephemeral=True)
        else:
            await interaction.response.send_message(
                "❌ 只有服務器擁有者或管理員可以重置統計數據！",
                ephemeral=True
            )
    except Exception as e:
        await interaction.response.send_message(f"❌ 重置時發生錯誤：{str(e)}", ephemeral=True)

@discord_client.tree.command(name="help", description="显示帮助信息")
async def help_command(interaction: discord.Interaction):
    """显示帮助信息"""
    try:
        help_text = """
💡 **AI助手指令列表**

基础命令：
• `/chat [消息]` - 与AI助手对话
• `/help` - 显示此帮助信息
• `/tokens` - 查看令牌使用统计
• `/cost` - 查看成本统计
• `/daily` - 查看每日统计
• `/info` - 查看机器人信息

管理命令（需要管理員權限）：
• `/sync` - 同步斜杠命令
• `/reset` - 重置统计数据
• `/config` - 查看当前配置
"""
        await interaction.response.send_message(help_text)
    except Exception as e:
        await interaction.response.send_message(f"❌ 显示帮助时发生错误：{str(e)}", ephemeral=True)

@discord_client.tree.command(name="daily", description="查看每日使用統計")
async def daily(interaction: discord.Interaction):
    """顯示每日使用統計"""
    if 'daily' not in token_usage:
        token_usage['daily'] = {}
    
    today = datetime.now().strftime('%Y-%m-%d')
    daily_stats = f"""
📅 **每日使用統計**
• 今日使用：{token_usage['daily'].get(today, 0):,} tokens
• 今日成本：${token_usage['daily'].get(f"{today}_cost", 0):.4f}

📊 **過去7天統計**
• 平均使用：{sum(token_usage['daily'].values()) / max(1, len(token_usage['daily'])):,.0f} tokens/天
• 平均成本：${sum(v for k, v in token_usage['daily'].items() if k.endswith('_cost')) / max(1, len([k for k in token_usage['daily'].keys() if k.endswith('_cost')])):,.4f}/天
"""
    await interaction.response.send_message(daily_stats)

@discord_client.tree.command(name="config", description="查看當前配置（僅管理員）")
async def config(interaction: discord.Interaction):
    """顯示當前配置"""
    if interaction.user.guild_permissions.administrator:
        cfg = get_config()
        config_text = f"""
⚙️ **當前配置**
• 模型：{cfg['model']}
• 最大文本：{cfg['max_text']:,} 字符
• 最大圖片：{cfg['max_images']} 張
• 最大消息：{cfg['max_messages']} 條
• 初始餘額：${cfg['billing']['initial_balance_usd']:.2f}
• 匯率：{cfg['billing']['exchange_rate']}
"""
        await interaction.response.send_message(config_text)
    else:
        await interaction.response.send_message("❌ 只有管理員可以查看配置！", ephemeral=True)

# 修改 sync 命令
@discord_client.tree.command(name="sync", description="同步斜杠命令（僅管理員）")
async def sync(interaction: discord.Interaction):
    """同步斜杠命令（僅管理員）"""
    try:
        await interaction.response.defer(ephemeral=True)
        
        # 检查是否为服务器所有者或管理员
        if interaction.guild and (interaction.user.id == interaction.guild.owner_id or interaction.user.guild_permissions.administrator):
            try:
                synced = await discord_client.tree.sync()
                await interaction.followup.send(
                    f"✅ 成功同步 {len(synced)} 個斜杠命令！",
                    ephemeral=True
                )
                logging.info(f"Synced {len(synced)} commands")
            except discord.HTTPException as e:
                await interaction.followup.send(
                    f"❌ 同步命令時發生錯誤：{str(e)}",
                    ephemeral=True
                )
        else:
            await interaction.followup.send(
                "❌ 只有服務器擁有者或管理員可以使用此命令！",
                ephemeral=True
            )
    except Exception as e:
        logging.exception("Error in sync command")
        await interaction.followup.send(f"❌ 執行同步時發生錯誤：{str(e)}", ephemeral=True)

# 添加一个全局错误处理器
@discord_client.event
async def on_interaction_error(interaction: discord.Interaction, error: app_commands.AppCommandError):
    if isinstance(error, app_commands.CommandOnCooldown):
        await interaction.response.send_message(
            f"⏳ 請稍等 {error.retry_after:.2f} 秒後再試！",
            ephemeral=True
        )
    elif isinstance(error, app_commands.MissingPermissions):
        await interaction.response.send_message(
            "❌ 你沒有權限使用此命令！",
            ephemeral=True
        )
    else:
        logging.error(f"Interaction error: {str(error)}")
        await interaction.response.send_message(
            f"❌ 執行命令時發生錯誤：{str(error)}",
            ephemeral=True
        )

# 修改處理消息的函數定義
async def process_message(message_obj, content, images=None):
    """處理消息和圖片"""
    cfg = get_config()
    provider, model = cfg["model"].split("/", 1)
    base_url = cfg["providers"][provider]["base_url"]
    api_key = cfg["providers"][provider].get("api_key", "sk-no-key-required")
    openai_client = AsyncOpenAI(base_url=base_url, api_key=api_key)

    # 構建消息
    messages = []
    
    # 處理圖片
    if images:
        content_parts = []
        for image in images:
            if image.content_type and "image" in image.content_type:
                image_data = await image.read()
                image_base64 = b64encode(image_data).decode('utf-8')
                content_parts.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{image.content_type};base64,{image_base64}"
                    }
                })
        
        if content:
            content_parts.append({
                "type": "text",
                "text": content
            })
        
        messages.append({
            "role": "user",
            "content": content_parts
        })
    else:
        messages.append({
            "role": "user",
            "content": content
        })

    if system_prompt := cfg["system_prompt"]:
        messages.append({"role": "system", "content": system_prompt})

    # 生成回复
    try:
        response_content = ""
        kwargs = dict(
            model=model, 
            messages=messages[::-1], 
            stream=True,
            max_tokens=cfg["extra_api_parameters"].get("max_tokens", 4096)
        )
        
        async for chunk in await openai_client.chat.completions.create(**kwargs):
            if chunk.choices[0].delta.content:
                response_content += chunk.choices[0].delta.content
                
        return response_content
                
    except Exception as e:
        logging.exception("Error processing message with image")
        return f"❌ 處理圖片時發生錯誤：{str(e)}"

async def show_usage_stats(interaction: discord.Interaction):
    """顯示基本使用統計信息"""
    total = token_usage['completion_tokens'] + token_usage['prompt_tokens']
    
    # 计算成本
    cfg = get_config()
    model = cfg["model"]
    cost = 0
    if model in MODEL_PRICES:
        prompt_cost = (token_usage['prompt_tokens'] / 1000) * MODEL_PRICES[model]['prompt']
        completion_cost = (token_usage['completion_tokens'] / 1000) * MODEL_PRICES[model]['completion']
        cost = prompt_cost + completion_cost

    # 获取账户余额
    billing = cfg.get('billing', {})
    initial_usd = billing.get('initial_balance_usd', 100.00)
    exchange_rate = billing.get('exchange_rate', 7.20)
    
    remaining_usd = initial_usd - token_usage.get('total_cost', 0) - cost
    remaining_rmb = remaining_usd * exchange_rate

    # 获取今日使用量
    today = datetime.now().strftime('%Y-%m-%d')
    if 'daily' not in token_usage:
        token_usage['daily'] = {}
    
    daily_tokens = token_usage['daily'].get(today, 0)

    usage_str = f"""
📊 **令牌使用概覽**

💬 **總計使用**
• 總令牌：{total:,} 個
• 總成本：${cost:.4f}

📅 **今日使用**
• 今日令牌：{daily_tokens:,} 個

💳 **餘額**
• USD：${remaining_usd:.2f}
• RMB：¥{remaining_rmb:.2f}
"""
    await interaction.response.send_message(usage_str)

# 添加错误处理装饰器
@discord_client.event
async def on_command_error(ctx, error):
    if isinstance(error, commands.errors.CommandNotFound):
        return
    logging.error(f"Command error: {str(error)}")

# 添加 info 命令
@discord_client.tree.command(name="info", description="查看機器人信息")
async def info(interaction: discord.Interaction):
    """顯示機器人信息"""
    try:
        # 计算运行时间
        uptime = datetime.now() - BOT_START_TIME
        days = uptime.days
        hours, remainder = divmod(uptime.seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        
        # 获取配置信息
        cfg = get_config()
        bot_info = cfg.get('bot_info', {})
        
        # 构建信息文本
        info_text = f"""
🤖 **AI助手信息**

⏱️ **運行時間**
• {days}天 {hours}小時 {minutes}分鐘 {seconds}秒

📊 **版本信息**
• 當前版本：{BOT_VERSION}
• {FORK_INFO}

💡 **主要功能**
"""
        # 添加功能列表
        features = bot_info.get('features', [
            "支持多種 LLM 模型",
            "支持圖片分析",
            "支持中英文對話",
            "支持多輪對話",
            "支持文本文件處理"
        ])
        
        for feature in features:
            info_text += f"• {feature}\n"
            
        info_text += f"""
🔧 **技術細節**
• 使用模型：{cfg['model']}
• 最大文本：{cfg['max_text']:,} 字符
• 最大圖片：{cfg['max_images']} 張
• 最大對話：{cfg['max_messages']} 條
"""
        
        await interaction.response.send_message(info_text)
    except Exception as e:
        logging.exception("Error in info command")
        await interaction.response.send_message(f"❌ 獲取信息時發生錯誤：{str(e)}", ephemeral=True)

# 添加启动时间记录
BOT_START_TIME = datetime.now()
BOT_VERSION = "v0.001"
FORK_INFO = "Fork from jakobdylanc/llmcord"

asyncio.run(main())
