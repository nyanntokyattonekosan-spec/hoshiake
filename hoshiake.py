# AI学習設定
import os
import asyncio
import json
import time
from datetime import datetime, timedelta
from typing import Dict, Tuple, List

import discord
from discord.ext import commands

# transformers / peft 等は任意。実行環境に合わせてインストール
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel  # if using LoRA adapters

# ---------- 設定 ----------
DISCORD_TOKEN = os.environ.get("DISCORD_TOKEN", "YOUR_TOKEN_HERE")
BASE_MODEL = os.environ.get("BASE_MODEL", "cyberagent/open-calm-3b")  # 例
ADAPTER_PATH = os.environ.get("INITIAL_ADAPTER", "adapters/lora-stable")
DATA_DIR = "data"
TRAIN_SCRIPT = "sft_incremental.py"  # 学習を行うスクリプト（subprocessで呼ぶ）
TRAIN_CHECK_INTERVAL = 60 * 5  # 5分ごとに学習トリガーチェック
CONV_TIMEOUT_SECONDS = 60 * 10  # 会話を続ける時間（例：10分）
MAX_HISTORY_TOKENS = 1024

# LoRA等の hot-swap 用ロック
model_lock = threading.RLock()

# ---------- グローバル状態 ----------
_loaded_once = False
tokenizer = None
model = None

# 会話保持（channel_id, user_id) -> list of (role, text)
active_conversations: Dict[Tuple[int, int], Dict] = {}
# 例: active_conversations[(channel.id, user.id)] = {
#    "expires": datetime,
#    "history": [ ("user", "..."), ("assistant","...") ]
# }

intents = discord.Intents.default()
intents.message_content = True
intents.messages = True

bot = commands.Bot(command_prefix="!", intents=intents)

# ---------- ユーティリティ: データ読み込み / フォーマット ----------
def load_jsonl(path):
    out = []
    if not os.path.exists(path):
        return out
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except:
                continue
    return out

def truncate_history_by_tokens(history: List[Tuple[str,str]], tokenizer, max_tokens=MAX_HISTORY_TOKENS):
    # シンプル実装：後ろから足していってトークン数超えたら切る
    total = 0
    rev = []
    for role, text in reversed(history):
        l = len(tokenizer(text)["input_ids"])
        if total + l > max_tokens:
            break
        rev.append((role, text))
        total += l
    return list(reversed(rev))

# ---------- モデルロード（on_readyで一度） ----------
def build_model_and_tokenizer(adapter_path: str = None, device=None):
    global tokenizer, model
    # device自動選択
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # 軽量化の例: 8-bitロードを使えるなら使う（環境依存）
    # from transformers import AutoModelForCausalLM
    # model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, load_in_8bit=True, device_map="auto")
    # ここでは通常ロード + PeftModel を使った例を示す
    print(f"[model] loading base model {BASE_MODEL} on {device} ...")
    base = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.float16 if device=="cuda" else torch.float32, device_map="auto" if device=="cuda" else None)
    tokenizer_local = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)
    if adapter_path and os.path.exists(adapter_path):
        print(f"[model] loading adapter from {adapter_path}")
        model_local = PeftModel.from_pretrained(base, adapter_path, device_map="auto" if device=="cuda" else None)
    else:
        model_local = base

    model_local.eval()
    return tokenizer_local, model_local

# ---------- 生成（スレッドセーフ） ----------
def generate_reply(prompt: str, max_new_tokens=150, temperature=0.8, top_p=0.9):
    global tokenizer, model, model_lock
    with model_lock:
        # トークナイズ & 生成
        inputs = tokenizer(prompt, return_tensors="pt").to(next(model.parameters()).device)
        out = model.generate(**inputs,
                             max_new_tokens=max_new_tokens,
                             do_sample=True,
                             temperature=temperature,
                             top_p=top_p,
                             repetition_penalty=1.05,
                             pad_token_id=tokenizer.eos_token_id)
        text = tokenizer.decode(out[0], skip_special_tokens=True)
    # modelが出力全文を返すことがあるので、プロンプト以降のみ取り出す実装は必要
    # ここでは簡易に最後の生成を返す
    return text

# ---------- Prompt組み立て ----------
SYSTEM_PROMPT = (
    "あなたは日本語で親切かつフレンドリーなアシスタントです。"
    "丁寧な言葉遣いで、必要なら確認を求めてください。"
)

def build_chat_prompt(history: List[Tuple[str,str]], user_msg: str):
    # history: [("user", "..."), ("assistant","..."), ...]
    s = f"<|system|>\n{SYSTEM_PROMPT}\n"
    for role, text in history:
        tag = "user" if role=="user" else "assistant"
        s += f"<|{tag}|>\n{text}\n"
    s += f"<|user|>\n{user_msg}\n<|assistant|>\n"
    return s

# ---------- 非同期学習ワーカー（サブプロセスで安全に学習） ----------
async def trainer_worker():
    """
    定期的に data/inbox.jsonl 等をチェックし、所定条件を満たしたら
    サブプロセスで学習スクリプトを起動します。学習中はログ出力を流す。
    """
    print("[trainer] worker started")
    while True:
        try:
            # 簡易トリガー：inbox の行数がしきい値を超えたら学習
            inbox = load_jsonl(os.path.join(DATA_DIR, "inbox.jsonl"))
            if len(inbox) >= 500:  # しきい値は調整
                tag = datetime.utcnow().strftime("v%Y%m%d%H%M%S")
                cmd = ["python", TRAIN_SCRIPT, "--tag", tag]
                print(f"[trainer] launching training: {' '.join(cmd)}")
                # 非同期サブプロセス起動（イベントループをブロックしない）
                proc = await asyncio.create_subprocess_exec(*cmd,
                                                            stdout=asyncio.subprocess.PIPE,
                                                            stderr=asyncio.subprocess.STDOUT)
                # ログをリアルタイムに読む（任意）
                while True:
                    line = await proc.stdout.readline()
                    if not line:
                        break
                    print("[trainer][log]", line.decode().rstrip())
                rc = await proc.wait()
                print(f"[trainer] training finished rc={rc}")
                # 学習が終わったら adapters に新しい adapter が置かれる想定 → ホットスワップを検討
                # ここで最新 adapter を検査してロードする処理を呼べる
            else:
                # 定期待機
                await asyncio.sleep(TRAIN_CHECK_INTERVAL)
        except Exception as e:
            print("[trainer] exception:", e)
            await asyncio.sleep(60)

# ---------- Hot-swap adapter（簡易） ----------
def try_reload_adapter_if_new(adapter_path: str):
    """
    もし adapter_path に新しい adapter が置かれていればモデルを差し替える（簡易実装）
    """
    global tokenizer, model, model_lock
    if not os.path.exists(adapter_path):
        return False
    try:
        with model_lock:
            print(f"[hot-swap] reloading adapter {adapter_path} ...")
            base = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.float16, device_map="auto")
            tok = AutoTokenizer.from_pretrained(BASE_MODEL)
            new_model = PeftModel.from_pretrained(base, adapter_path, device_map="auto")
            new_model.eval()
            # スワップ
            tokenizer = tok
            model = new_model
        return True
    except Exception as e:
        print("[hot-swap] failed:", e)
        return False

# ---------- Discord イベント ----------
@bot.event
async def on_ready():
    global _loaded_once, tokenizer, model
    if _loaded_once:
        return
    print(f"[bot] logged in as {bot.user} (id={bot.user.id})")
    # 1回だけ行う初期化
    try:
        tokenizer, model = build_model_and_tokenizer(ADAPTER_PATH)
        print("[bot] model and tokenizer loaded")
    except Exception as e:
        print("[bot] model load failed:", e)
        # 失敗しても動作継続（ただし生成不可）
    # 起動後に非同期ワーカーを開始（イベントループで）
    bot.loop.create_task(trainer_worker())
    _loaded_once = True

@bot.event
async def on_message(message: discord.Message):
    # 自分自身やボットのメッセージは無視
    if message.author.bot:
        return

    # 応答条件:
    #  - ボットにメンションされている (message.mentions)
    #  - または (channel, user) が active_conversations にあって期限内
    key = (message.channel.id, message.author.id)
    now = datetime.utcnow()

    mentioned = bot.user in message.mentions
    active = False
    if key in active_conversations:
        entry = active_conversations[key]
        if entry["expires"] > now:
            active = True
        else:
            # 期限切れなら削除
            del active_conversations[key]

    if not (mentioned or active):
        # 無音モード（応答しない）
        return

    # 会話の履歴を準備（メモリに保存。実運用なら DB を推奨）
    history = []
    if active:
        history = active_conversations[key]["history"]
    else:
        # 新しく会話を始めるとき、過去数件のメッセージを拾う（簡易実装）
        # ここでは直近の発言1つを履歴に入れる例
        history = [("user", message.content)]

    # build prompt
    prompt = build_chat_prompt(history, message.content)

    # 生成（非同期に重い処理を回すため to_thread を使う）
    loop = asyncio.get_running_loop()
    try:
        generated = await loop.run_in_executor(None, generate_reply, prompt)
    except Exception as e:
        generated = "ごめんなさい、応答生成でエラーが起きました。"

    # 簡易 post-processing: 応答テキストを短くする等
    reply_text = generated.strip()
    if len(reply_text) > 1900:
        reply_text = reply_text[:1900] + "…"

    # 返信（メンションは省略してフレンドリーに）
    try:
        await message.channel.send(reply_text)
    except Exception as e:
        print("[bot] send fail:", e)

    # active_conversations を更新（継続させる）
    expiry = datetime.utcnow() + timedelta(seconds=CONV_TIMEOUT_SECONDS)
    new_history = history + [("user", message.content), ("assistant", reply_text)]
    # トークン長でカット
    try:
        new_history = truncate_history_by_tokens(new_history, tokenizer, MAX_HISTORY_TOKENS)
    except:
        pass
    active_conversations[key] = {"expires": expiry, "history": new_history}

# ---------- 起動 ----------
def main():
    # 追加: Render などで直接実行する際に使う
    bot.run(DISCORD_TOKEN)

if __name__ == "__main__":
    main()

