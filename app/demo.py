"""A short built-in class used by the guided tour: real transcript, summary and an offline 'AI' that
answers with prepared explanations, so a new user can try every feature without recording or a key."""
import re
import time

TITLE = "新手示例课 · 需求弹性"
COURSE = "示例课程"

LINES = [
    (0, "Good morning everyone, let's get started.", "大家早上好，我们开始上课。"),
    (4, "Today we're going to continue our discussion of supply and demand.", "今天我们继续讨论供给与需求。"),
    (9, "Last week we talked about the law of demand: when the price of a good goes up, the quantity demanded goes down.",
     "上周我们讲了需求定律：一种商品的价格上涨时，需求量会下降。"),
    (17, "Now I want to introduce a really important idea called price elasticity of demand.",
     "现在我要介绍一个非常重要的概念，叫作需求的价格弹性（price elasticity of demand）。"),
    (23, "Elasticity measures how sensitive the quantity demanded is to a change in price.",
     "弹性衡量的是需求量对价格变化有多敏感。"),
    (29, "If a small change in price leads to a big change in quantity, we say demand is elastic.",
     "如果价格的小变化带来需求量的大变化，我们就说需求是有弹性的（elastic）。"),
    (36, "If a big change in price only leads to a small change in quantity, demand is inelastic.",
     "如果价格的大变化只带来需求量的小变化，需求就是缺乏弹性的（inelastic）。"),
    (43, "Let me give you an example with gasoline.", "我举一个汽油的例子。"),
    (46, "When gas prices go up by ten percent, people still need to drive to work, so the quantity demanded only falls by maybe two or three percent.",
     "油价上涨百分之十时，人们还是要开车上班，所以需求量可能只下降百分之二到三。"),
    (56, "Compare that with a specific brand of cereal: if one brand raises its price, you can easily switch to another brand.",
     "再对比某个特定品牌的麦片：如果一个品牌涨价，你很容易换成别的牌子。"),
    (65, "The formula is the percentage change in quantity demanded divided by the percentage change in price.",
     "公式是：需求量变化的百分比除以价格变化的百分比。"),
    (72, "Why do businesses care? Because elasticity tells you what happens to total revenue when you change your price.",
     "企业为什么关心这个？因为弹性能告诉你，改变价格时总收入（total revenue）会怎样变化。"),
    (80, "If demand is inelastic, raising the price actually increases total revenue.",
     "如果需求缺乏弹性，涨价反而会增加总收入。"),
    (86, "The more close substitutes a good has, the more elastic its demand.",
     "一种商品的相近替代品（substitutes）越多，它的需求就越有弹性。"),
    (92, "For next week, please read chapter five, and remember the problem set is due Friday at five p.m.",
     "下周之前请阅读第五章，另外别忘了习题集周五下午五点前要交。"),
]

SUMMARY = """## 本节课主题
需求的价格弹性：需求量对价格变化有多敏感，以及它和企业总收入的关系。

## 主要内容
### 1. 什么是需求弹性
- 弹性衡量需求量对价格变化的敏感程度 [00:23]
- 价格小变化 → 需求量大变化：**有弹性**；价格大变化 → 需求量小变化：**缺乏弹性** [00:29]

### 2. 例子
- 汽油：涨价 10%，需求量只降 2–3%，缺乏弹性 [00:46]
- 某个品牌的麦片：很容易换别的牌子，弹性很大 [00:56]

### 3. 计算与应用
- 弹性 = 需求量变化百分比 ÷ 价格变化百分比 [01:05]
- 需求缺乏弹性时，涨价会让总收入增加 [01:20]
- 替代品越多，需求越有弹性 [01:26]

## 重要概念 / 术语
- 需求的价格弹性 — price elasticity of demand：需求量对价格变化的敏感程度
- 总收入 — total revenue：价格 × 数量
- 替代品 — substitutes：可以互相替换使用的商品

## 作业、考试与截止日期
- 阅读第五章 [01:32]
- 习题集周五下午 5 点截止 [01:32]

## 可能考到的点
- 根据弹性判断涨价后总收入的变化
- 用替代品多少解释弹性大小

## 复习建议
先记住公式，再用汽油和麦片两个例子各自说明一遍。
"""

TERMS = [
    {"term": "price elasticity of demand", "zh": "需求价格弹性", "en": "How much the quantity demanded changes when the price changes."},
    {"term": "elastic", "zh": "有弹性的", "en": "Quantity demanded changes a lot when the price changes (elasticity > 1)."},
    {"term": "inelastic", "zh": "缺乏弹性的", "en": "Quantity demanded changes little when the price changes (elasticity < 1)."},
    {"term": "total revenue", "zh": "总收入", "en": "Price times quantity sold."},
    {"term": "substitutes", "zh": "替代品", "en": "Goods that can be used in place of each other."},
    {"term": "law of demand", "zh": "需求定律", "en": "When price rises, quantity demanded falls, other things equal."},
    {"term": "quantity demanded", "zh": "需求量", "en": "The amount buyers want to buy at a given price."},
]

# prepared explanations, picked by what the user selected
ANSWERS = [
    (r"elastic|弹性|sensitive|敏感",
     "**需求的价格弹性**（price elasticity of demand）说的是：价格变一点，买的人会多买/少买多少。\n\n"
     "- 变化大 → **有弹性**，比如某个品牌的麦片，涨价了大家就换牌子 [00:56]\n"
     "- 变化小 → **缺乏弹性**，比如汽油，涨价了还是得开车上班 [00:46]\n\n"
     "记忆小窍门：**能不能轻松“不买”或“换一个”**，能就有弹性。"),
    (r"revenue|收入",
     "**总收入 = 价格 × 数量**。\n\n"
     "如果需求**缺乏弹性**，涨价后买的人减少得不多，所以总收入反而上升 [01:20]。"
     "这就是加油站、药品敢涨价的原因；反过来，需求有弹性的商品涨价往往会让收入下降。"),
    (r"substitut|替代",
     "**替代品**就是可以拿来代替的东西，比如不同品牌的麦片。\n\n"
     "替代品越多，消费者越容易“换一个”，所以需求越有弹性 [01:26]。"),
    (r"formula|percent|公式|百分比",
     "公式：**弹性 = 需求量变化的百分比 ÷ 价格变化的百分比** [01:05]\n\n"
     "例：油价涨 10%，需求量降 2.5%，弹性 = 2.5 ÷ 10 = 0.25，小于 1，所以**缺乏弹性**。"),
    (r"gas|汽油",
     "汽油是**缺乏弹性**的典型例子：涨价 10%，需求量只降 2–3% [00:46]，因为短期内大家没有别的办法去上班。"),
    (r"problem set|chapter|作业|习题|第五章",
     "老师布置了两件事 [01:32]：\n- 阅读**第五章**\n- **习题集周五下午 5 点**前交"),
]
GENERIC = ("这段话的意思是：{zh}\n\n"
           "放在这节课里看，它是在讲**需求弹性**的一部分：价格变化时，需求量变化得多还是少。"
           "可以点回答里的时间点 [00:23] 跳回原文复习。")
FOLLOWUP = ("好问题！简单说：**看消费者能不能轻松少买或换别的**。能 → 有弹性；不能 → 缺乏弹性。"
            "考试时常让你根据这一点判断涨价后总收入会怎么变 [01:20]。")
KEY = [(r"elastic|弹性|sensitive", "弹性 = 需求量对价格变化的敏感程度；能轻松换就有弹性"),
       (r"revenue|收入", "需求缺乏弹性时涨价，总收入反而增加"),
       (r"substitut|替代", "替代品越多，需求越有弹性"),
       (r"formula|percent|公式", "弹性 = 需求量变化% ÷ 价格变化%")]


def segments():
    return [{"id": i, "t": float(t), "en": en, "zh": zh} for i, (t, en, zh) in enumerate(LINES)]


def answer(quote, question, first):
    """(text) of the prepared reply."""
    if not first and question:
        return FOLLOWUP
    text = (quote or question or "").lower()
    for pat, a in ANSWERS:
        if re.search(pat, text):
            return a
    zh = ""
    for _, en, z in LINES:
        if quote and quote.lower()[:30] in en.lower():
            zh = z
            break
    return GENERIC.format(zh=zh or "老师在解释这一部分的概念。")


def key_point(thread):
    text = (thread.get("anchor") or {}).get("text", "") + " " + " ".join(t.get("a", "") for t in thread.get("turns", []))
    for pat, k in KEY:
        if re.search(pat, text, re.I):
            return k
    return "示例笔记：价格变化时需求量变多少"


def stream(text, on_text, pace=0.012):
    """Hand out the reply a few characters at a time, like a real answer."""
    out = ""
    for i in range(0, len(text), 4):
        out = text[:i + 4]
        on_text(out)
        time.sleep(pace)
    return text
