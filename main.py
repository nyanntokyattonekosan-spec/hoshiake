# main.py
import os
import asyncio
import threading
import logging
import time
from functools import partial

from flask import Flask, jsonify
import discord
from discord.ext import commands

# model.py から TransformerModel, CharTokenizer, train_on_texts, mask_pii を import します
from model import TransformerModel, CharTokenizer, train_on_texts, load_tokenizer_if_exists, load_model_if_exists, mask_pii

# --- 設定 ---
DISCORD_TOKEN = os.environ.get("DISCORD_TOKEN")
TARGET_CHANNEL_ID = int(os.environ.get("TARGET_CHANNEL_ID", "0"))
PORT = int(os.environ.get("PORT", 5000))
RETRAIN_INTERVAL_SECONDS = int(os.environ.get("RETRAIN_INTERVAL_SECONDS", 60*60))  # 1時間毎デフォルト
MAX_CONTEXT_MESSAGES = 6

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("bot")

# --- Flask アプリ（Renderのヘルスチェック用） ---
app = Flask(__name__)

@app.route("/health")
def health():
    return jsonify({"status":"ok"})

@app.route("/")
def index():
    return "Discord VTuber Bot running."

def run_flask():
    # Render は PORT 環境変数を使う
    app.run(host="0.0.0.0", port=PORT)

# --- Discord Bot セットアップ ---
intents = discord.Intents.default()
intents.message_content = True
intents.messages = True
bot = commands.Bot(command_prefix="!", intents=intents)

# --- モデル読み込み（軽量プロトタイプ） ---
device = "cuda" if os.environ.get("USE_CUDA","0") == "1" else "cpu"
tokenizer = load_tokenizer_if_exists("data/tokenizer.json") or CharTokenizer()
model = load_model_if_exists("data/model.pt") or TransformerModel(vocab_size=tokenizer.vocab_size)

# move model to device
model.to(device)
model.eval()

# --- ヘルパ：直近の会話を取得してプロンプト化 ---
async def build_context_prompt(channel, author, limit=MAX_CONTEXT_MESSAGES):
    msgs = []
    async for m in channel.history(limit=limit, oldest_first=False):
        if m.author.bot:
            continue
        msgs.append(f"{m.author.display_name}: {m.content}")
    # reverse to chronological
    msgs = list(reversed(msgs))
    # persona の一文を先頭に置く
    persona = "あなたは女性Vtuber風の明るく親しみやすいキャラクターです。"
    prompt = persona + "\n" + "\n".join(msgs) + f"\n{author.display_name}:"
    return prompt

# --- メッセージ受信ハンドラ ---
@bot.event
async def on_message(message):
    if message.author.bot:
        return
    if message.channel.id != TARGET_CHANNEL_ID:
        return

    # build prompt from recent messages
    prompt = await build_context_prompt(message.channel, message.author)
    prompt = mask_pii(prompt)
    # tokenize
    token_ids = tokenizer.encode(prompt, add_bos=True, add_eos=False)
    # generation (simple greedy / top-k)
    try:
        # generate is a CPU/GPU function; run in executor if it's blocking heavy work
        loop = asyncio.get_running_loop()
        gen_ids = await loop.run_in_executor(None, partial(
            model.generate, token_ids, max_new_tokens=120, temperature=0.9, top_k=40, device=device))
        reply = tokenizer.decode(gen_ids)
        await message.channel.send(reply)
    except Exception as e:
        logger.exception("Generation failed")
        await message.channel.send("ごめんなさい、ちょっと調子が悪いみたい。")

# --- バックグラウンド：新着メッセージを定期取得して短時間再学習 ---
async def retrain_loop():
    await bot.wait_until_ready()
    channel = bot.get_channel(TARGET_CHANNEL_ID)
    if channel is None:
        logger.warning("Target channel not found on start.")
    last_seen = None

    while not bot.is_closed():
        try:
            new_texts = []
            async for m in channel.history(limit=200, after=last_seen, oldest_first=True):
                if m.author.bot:
                    continue
                cleaned = mask_pii(m.content)
                if cleaned.strip():
                    new_texts.append(cleaned)
                last_seen = m.created_at
            if new_texts:
                logger.info(f"Found {len(new_texts)} new messages, starting background training.")
                # training is blocking CPU/GPU work -> run in executor
                loop = asyncio.get_running_loop()
                # train_on_texts is a function in model.py that performs limited training and saves checkpoint
                await loop.run_in_executor(None, partial(train_on_texts, new_texts, tokenizer, "data/model.pt"))
                logger.info("Background training finished.")
            await asyncio.sleep(RETRAIN_INTERVAL_SECONDS)
        except Exception:
            logger.exception("Error in retrain_loop")
            await asyncio.sleep(60)

# --- 起動処理 ---
def start_bot():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    # start background task after bot ready
    loop.create_task(retrain_loop())
    loop.run_until_complete(bot.start(DISCORD_TOKEN))

if __name__ == "__main__":
    # start flask in separate thread (so Render's web port is served)
    t = threading.Thread(target=run_flask, daemon=True)
    t.start()
    # start discord bot (blocking)
    start_bot()

