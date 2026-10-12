"""Critique prompts, segmentation, critique parsing and the math grader pool of CriticOPD.

Copied verbatim from the training code (relay-opd/verl/experimental/agent_loop/critic_opd_agent_loop.py,
module-level part before the agent loop class), so that the diagnosis uses exactly the prompts and parsers
of training. The only change is the location of the grader package inside _GRADER_SRC. Do not edit.
"""
import asyncio
import json
import logging
import os
import random
import re
from typing import Any
from uuid import uuid4

os.environ.setdefault("CRITICOPD_DIAG_PKG_PARENT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

_SEP = re.compile(r"\n\s*\n")
_LEADIN = re.compile(r"^\s*(?:#{1,6}\s|[-*_=]{3,}\s*$|\*\*[^*]{0,40}\*\*\s*:?\s*$"
                     r"|(?:So|Now|Then|Next|Thus|Hence|Therefore|Step\s*\d+|Case\s*\d+)\b[^.\n]{0,30}:\s*$)", re.I)
_RE_SEG = re.compile(r"SEGMENT:\s*\[?(\d+)", re.I)
_RE_Q = re.compile(r"QUOTE:\s*(.+?)(?=\n\s*FEEDBACK:|\Z)", re.I | re.S)
_RE_FB = re.compile(r"FEEDBACK:\s*(.+)", re.I | re.S)

CRITIC_SYS = (
    "You are a meticulous math teacher reviewing a student's solution. "
    "The student's FINAL ANSWER IS WRONG. Find the FIRST place where the reasoning goes wrong.\n\n"
    "Rules:\n"
    "1. Identify exactly ONE segment -- the FIRST containing a genuine error.\n"
    "2. Quote the exact wrong sentence verbatim.\n"
    "3. Say briefly what is wrong and what to do instead.\n"
    "4. DO NOT continue the solution. DO NOT compute or state the final answer.\n\n"
    "Reply exactly as:\nSEGMENT: <index>\nQUOTE: <verbatim>\nFEEDBACK: <what is wrong and what to do>"
)
# ---- GT 思路(CRITIC_OPD_GIVE_GT=1):critic 拿到正确答案 ----
# 原 prompt 只告诉 teacher「学生答错了」,不给正确答案,teacher 必须先自己解出题目才能判断
# 哪一步错 —— 而 Qwen3-4B 在 aime24 上自己也只有约 40 分。这里把正确答案写进任务定义,
# 给出定位方法,排除「学生后来自己纠正的弯路」这类误判,并允许 teacher 说「找不到」:
# 错误的定位比没有定位更糟,它会让学生从一个本来正确的位置重写。
_GT_RULES = (
    "How to locate the error:\n"
    "Read the numbered segments in order. For each one ask: given everything before it, is this "
    "step still correct, and does it still allow the problem to be solved? The FIRST segment "
    "where the answer is no is the error.\n\n"
    "Do NOT flag a segment merely because it is: an exploratory attempt the student later "
    "abandons or corrects; a longer route than necessary; or sloppy notation that does not change "
    "the mathematics. Flag only a step that is actually wrong.\n\n"
    "Rules you must follow:\n"
    "1. Identify exactly ONE segment -- the FIRST one containing a genuine error.\n"
    "2. Quote verbatim the single sentence in that segment that carries the error.\n"
    "3. Feedback: say what is wrong and what to do from that point instead. The student will "
    "resume writing immediately after this point, so phrase it as guidance they can act on. "
    "Two or three sentences at most.\n"
    "4. NEVER reveal the final answer. Do not state it, do not write an expression that "
    "evaluates to it, do not say what it should be. Describe the method, not the result.\n"
    "5. If every step looks correct to you and you cannot identify a genuine error, reply with "
    "SEGMENT: NONE. Do not guess -- a wrong location is worse than none.\n\n"
    "Reply in exactly this format:\n"
    "SEGMENT: <index, or NONE>\n"
    "QUOTE: <the exact wrong sentence, copied verbatim>\n"
    "FEEDBACK: <what is wrong and what to do instead>"
)
CRITIC_SYS_GT = (
    "You are a meticulous math teacher. A student has submitted a complete solution whose "
    "FINAL ANSWER IS WRONG. You are given the correct answer. Use it as a reference to trace the "
    "student's work forward and find the EARLIEST step after which the correct answer can no "
    "longer be reached.\n\n" + _GT_RULES)
# 写不完(顶上限)的轨迹没有最终答案,不能说「答案错了」;让 teacher 判断这条路还走不走得通。
CRITIC_SYS_GT_CAPPED = (
    "You are a meticulous math teacher. A student's solution ran on without ever reaching an "
    "answer. You are given the correct answer. Use it as a reference to trace the student's work "
    "forward and find the EARLIEST step that sent the reasoning off track. If the work is still on "
    "a valid path and was merely slow, reply SEGMENT: NONE.\n\n" + _GT_RULES)
_RE_NONE = re.compile(r"SEGMENT:\s*\[?\s*(NONE|N/A|-)\b", re.I)

# ---- 参考解答版批改(CRITIC_OPD_REF_FILE):critic 额外拿到 teacher 离线做对的一份完整解答 ----
# 离线检验(criticopd/ref_probe.py,400 条同批错题):修复成功率 22.0% -> 36.0%,
# 95% CI [+9.8, +18.1];严格口径的答案泄露不变(9.2% vs 9.2%),重写不变短,误报减少。
# 提示词与 ref_probe.py 的 B 逐字一致。
_REF_RULES = (
    "How to locate the error:\n"
    "Read the student's numbered segments in order. Check the mathematics of each one, using the "
    "reference solution as a guide to what is true. The FIRST segment containing a genuinely wrong "
    "step -- a false statement, a wrong computation, an invalid deduction, or a wrong setup -- is "
    "the error.\n\n"
    "The student may solve the problem differently from the reference. A different approach is NOT "
    "an error: flag a segment only if its mathematics is actually wrong, never merely because it "
    "differs from the reference. Also do NOT flag an exploratory attempt the student later abandons "
    "or corrects, a longer route than necessary, or sloppy notation that does not change the "
    "mathematics.\n\n"
    "Rules you must follow:\n"
    "1. Identify exactly ONE segment -- the FIRST one containing a genuine error.\n"
    "2. Quote verbatim the single sentence in that segment that carries the error.\n"
    "3. Feedback: say concretely what is wrong at that step and what the correct reasoning for THAT "
    "step is (you may use the reference to state the correct method or fact for that step). The "
    "student will resume writing immediately after this point. Two or three sentences at most.\n"
    "4. NEVER reveal the final answer. Do not describe later steps of the reference and do not copy "
    "it.\n"
    "5. If every step looks correct to you, reply SEGMENT: NONE. Do not guess -- a wrong location is "
    "worse than none.\n\n")
_REF_FMT = ("Reply in exactly this format:\n"
            "SEGMENT: <index, or NONE>\n"
            "QUOTE: <the exact wrong sentence, copied verbatim>\n"
            "FEEDBACK: <what is wrong and what to do instead>")
CRITIC_SYS_REF = (
    "You are a meticulous math teacher. A student has submitted a complete solution whose FINAL "
    "ANSWER IS WRONG. You are given the correct final answer and a correct reference solution. "
    "Use them to find the EARLIEST step where the student's work goes wrong.\n\n" + _REF_RULES + _REF_FMT)
# 写不完的轨迹:离线检验里只有 2.2%,没有单独测;开头沿用 CRITIC_SYS_GT_CAPPED 的说法,其余与上面相同。
CRITIC_SYS_REF_CAPPED = (
    "You are a meticulous math teacher. A student's solution ran on without ever reaching an answer. "
    "You are given the correct final answer and a correct reference solution. Use them to find the "
    "EARLIEST step that sent the reasoning off track. If the work is still on a valid path and was "
    "merely slow, reply SEGMENT: NONE.\n\n" + _REF_RULES + _REF_FMT)

# ---- 截断思路(CRITIC_OPD_TRUNC=1):学生改完出错的那一步就停 ----
# 原设计让学生从拼接点一直写到 EOS:离线 67.8% 的重写顶上限、修复成功率 0.0%,却全部以
# mask=1 进了训练序列。出错段本身中位只有 149 字符(约 40 token),所以只训练「改这一步」。
# 边界 = 有实质内容之后的第一个空行(和分段规则一致)。
_BLANK = re.compile(r"\S[^\S\n]*\n[^\S\n]*\n")


def _cut_at_blank_line(tok, ids: list[int]) -> int:
    """返回 k,使 ids[:k] 恰好结束在「实质内容之后的第一个空行」处(含该空行)。
    「前缀里是否已出现空行」对 k 单调,所以二分;找不到空行就整段保留。"""
    if not ids or not _BLANK.search(tok.decode(ids, skip_special_tokens=True)):
        return len(ids)
    lo, hi = 1, len(ids)
    while lo < hi:
        mid = (lo + hi) // 2
        if _BLANK.search(tok.decode(ids[:mid], skip_special_tokens=True)):
            hi = mid
        else:
            lo = mid + 1
    return lo


FEEDBACK_TMPL = ("\n\n[Teacher feedback] {fb}\n\nUsing this feedback, continue the solution from here. "
                 "Put your final answer in \\boxed{{}}.\n\n")
# 基线 Critique-GRPO 的精炼指令,与官方实现(critique-GRPO/verl/critique_grpo/refinement_prompts.py)相同:
# 新开一轮对话,user 消息里给出题目、原解答和批改,要求逐步推理重答。
# (2026-10-10 冒烟:把批改接在原回答后面、在同一个 assistant 轮里续写时,学生只补一行最终答案,精炼退化成抄答案)
CGRPO_REFINE_SYS = "You are a helpful assistant."
CGRPO_REFINE_USER = (
    "Given the following inputs:\n"
    "**Question**: {question}\n"
    "**Previous Solution**: {failed}\n"
    "**Critique**: {critique}\n\n"
    "Please re-answer by:\n"
    "- Correcting potential errors identified in the critique, if they exist.\n"
    "- Providing clear, step-by-step reasoning.\n"
    "- Formatting the final answer as `\\boxed{{answer}}`.\n\n"
    "Ensure the revised solution addresses all issues raised in the critique.")


def _is_leadin(p):
    return bool(_LEADIN.match(p)) or (len(p) <= 12 and p.rstrip().endswith(":"))


def segment(text):
    """按空行切,再把标题/分隔线/引导句并入后一段。按形态判断而不是长度 ——
    只按"短"合并会把正常的短内容块(一行公式、一句结论)也并掉。"""
    parts = [p.strip() for p in _SEP.split(text) if p.strip()]
    merged, buf = [], []
    for p in parts:
        if _is_leadin(p):
            buf.append(p); continue
        merged.append("\n\n".join(buf + [p]) if buf else p); buf = []
    if buf: merged.append("\n\n".join(buf))
    return merged or parts


def parse_critic(text, segs):
    """优先用引用在原文里定位,编号只作兜底 —— critic 的编号经常偏一两格,引用不会。"""
    if _RE_NONE.search(text):          # critic 明确说找不到错:放弃,不让引用匹配绕过
        return None, None
    mq, mf, ms = _RE_Q.search(text), _RE_FB.search(text), _RE_SEG.search(text)
    fb = mf.group(1).strip() if mf else None
    quote = mq.group(1).strip().strip('"') if mq else None
    idx = None
    if quote and len(quote) >= 8:
        norm = lambda t: re.sub(r"\s+", " ", t).strip()
        nq = norm(quote)[:80]
        for i, s in enumerate(segs):
            if quote[:60] in s or nq in norm(s):
                idx = i; break
    if idx is None and ms:
        j = int(ms.group(1))
        if 0 <= j < len(segs): idx = j
    return idx, fb



# ---- k 消融(CRITIC_OPD_KERR):批改老师按顺序列出所有错误,在第 k 个错误处断开 ----
# 规则与 CRITIC_SYS_GT 相同,只把「只找第一个」改成「按顺序列出全部(至多 5 个)」。
# k=1 时批语格式与 R4GT 完全一样(只含第一个错误的反馈),所以 k=1 与 R4GT 只差批改提示词本身。
_MULTI_RULES = (
    "How to locate the errors:\n"
    "Read the numbered segments in order. For each one ask: given everything before it, is this step "
    "correct? List every segment that contains a genuine error, in the order in which they appear.\n\n"
    "Do NOT flag a segment merely because it is: an exploratory attempt the student later abandons or "
    "corrects; a longer route than necessary; or sloppy notation that does not change the mathematics. "
    "Flag only a step that is actually wrong.\n\n"
    "Rules you must follow:\n"
    "1. List the erroneous segments in order of appearance, at most 5.\n"
    "2. For each one, quote verbatim the single sentence that carries the error.\n"
    "3. For each one, say what is wrong and what to do instead. The student will resume writing after "
    "the last listed error, so phrase it as guidance they can act on. Two or three sentences per error.\n"
    "4. NEVER reveal the final answer. Do not state it, do not write an expression that evaluates to it, "
    "do not say what it should be. Describe the method, not the result.\n"
    "5. If every step looks correct to you and you cannot identify a genuine error, reply with "
    "SEGMENT: NONE. Do not guess -- a wrong location is worse than none.\n\n"
    "Reply in exactly this format, one block per error:\n"
    "ERROR 1\nSEGMENT: <index>\nQUOTE: <the exact wrong sentence, copied verbatim>\n"
    "FEEDBACK: <what is wrong and what to do instead>\n"
    "ERROR 2\nSEGMENT: <index>\nQUOTE: <...>\nFEEDBACK: <...>")
CRITIC_SYS_MULTI = (
    "You are a meticulous math teacher. A student has submitted a complete solution whose FINAL ANSWER "
    "IS WRONG. You are given the correct answer. Use it as a reference to trace the student's work "
    "forward and find the steps that contain genuine errors.\n\n" + _MULTI_RULES)
CRITIC_SYS_MULTI_CAPPED = (
    "You are a meticulous math teacher. A student's solution ran on without ever reaching an answer. "
    "You are given the correct answer. Use it as a reference to trace the student's work forward and find "
    "the steps that sent the reasoning off track. If the work is still on a valid path and was merely "
    "slow, reply SEGMENT: NONE.\n\n" + _MULTI_RULES)
# 消融 CRITIC_OPD_FB_MODE=generic:与方法同一模板,只放一条不含内容的反馈
GENERIC_FEEDBACK = "Your solution contains errors. Check the previous reasoning and redo it."
# 消融 CRITIC_OPD_ALL 用:答对的 rollout 不能说「答案是错的」,第一段改成中性描述,其余规则不变
CRITIC_SYS_MULTI_ANY = (
    "You are a meticulous math teacher. A student has submitted a complete solution. You are given the correct "
    "answer. Use it as a reference to trace the student's work forward and find the steps that contain genuine "
    "errors.\n\n" + _MULTI_RULES)
_RE_ERRHDR = re.compile(r"\bERROR\s*\d+\s*:?\s*$", re.I)


def parse_critic_multi(text, segs, max_err=5):
    """解析多错误批改 -> [(段号, 引用, 反馈)],按段号升序、去重。
    按「SEGMENT:」切块(批改老师把序号和段号写在同一行时也能切开),每块复用 parse_critic 的
    引用优先定位;块尾残留的下一个「ERROR k」标题从反馈里去掉。整段只有 NONE 时返回空列表。"""
    errs, seen = [], set()
    for b in re.split(r"(?i)(?=SEGMENT\s*:)", text):
        if not re.match(r"(?i)SEGMENT\s*:", b):
            continue
        idx, fb = parse_critic(b, segs)
        if idx is None or not fb or idx in seen:
            continue
        fb = "\n".join(ln for ln in fb.strip().splitlines() if not _RE_ERRHDR.search(ln)).strip()
        if not fb:
            continue
        mq = _RE_Q.search(b)
        q = mq.group(1).strip().strip('"') if mq else ""
        errs.append((idx, q, fb)); seen.add(idx)
    errs.sort(key=lambda e: e[0])
    return errs[:max_err]


_RE_LAST_BLOCK = re.compile(r"(?i)(?:ERROR\s*\d+\s*:?\s*)?SEGMENT\s*:")


def drop_incomplete_tail(text):
    """批改输出撞到长度上限时,最后一块(从最后一个 SEGMENT: 起)一定没写完:整块去掉。
    只在确认被截断时调用;没有任何 SEGMENT: 时原样返回。"""
    ms = list(_RE_LAST_BLOCK.finditer(text))
    return text[:ms[-1].start()] if ms else text


def kerr_select(errs, kerr):
    """在第 k 个错误处断开;错误数少于 k 时取最后一个。kerr 为 '1'/'2'/'3'/.../'last'。"""
    if not errs:
        return []
    k = len(errs) if str(kerr) == "last" else min(int(kerr), len(errs))
    return errs[:k]


def leaks_answer(fb, gt, stu):
    """反馈里出现了标准答案(独立成词),而学生原文里没有 -> 视为泄露答案。"""
    if gt is None:
        return False
    g = str(gt).strip().strip("$")
    if not g:
        return False
    # 前后不能紧挨字母数字或小数点+数字(避免 3280、1.328、23.5 误判);句末的「23.」要算
    pat = r"(?<![\w.])" + re.escape(g) + r"(?!\w|\.\d)"
    return bool(re.search(pat, fb or "")) and not re.search(pat, stu or "")


def drop_leaky(errs, gt, stu):
    """去掉反馈泄露答案的错误。多错误批改越往后越容易在反馈里写出答案(k=1/2/last 约 1.5%/7%/10%),
    不去掉的话 k 越大重写越容易「照抄答案」,k 的比较就不公平。"""
    return [e for e in errs if not leaks_answer(e[2], gt, stu)]


def kerr_feedback(sel):
    """k=1 与 R4GT 的批语格式完全一样;k>1 时按顺序列出各个错误(附原文引用,截到 200 字)。"""
    if len(sel) == 1:
        return sel[0][2]
    parts = [f'({i + 1}) In "{q[:200]}": {fb}' for i, (_, q, fb) in enumerate(sel)]
    return f"Your solution contains {len(sel)} errors.\n" + "\n".join(parts)


def _tok_prefix_for_text(tok, ids: list[int], text: str) -> int:
    """返回最小的 k,使 decode(ids[:k]) 已经覆盖 text。

    之前是把保留段 decode 成文本再 encode 回 token,这会引入 tokenization 漂移:
    训练序列不再是学生自己轨迹的前缀,而且 out.log_probs 和 kept_ids 对不上 ——
    rollout_log_probs 在 bypass 模式下会直接变成 old_log_probs(PPO 比值的分母),
    错位就会污染训练。直接在原始 rollout token 上二分切,两个问题一起消掉。
    """
    target = len(text.rstrip())
    if target <= 0:
        return 0
    lo, hi = 0, len(ids)
    while lo < hi:
        mid = (lo + hi) // 2
        if len(tok.decode(ids[:mid], skip_special_tokens=True).rstrip()) >= target:
            hi = mid
        else:
            lo = mid + 1
    return lo


# ---- 进程内判分 -> 独立进程判分 ----
# 2026-10-02 R4GTV(13949875)在 step 2 整体冻住 47 分钟:判分线程卡在 sympy 的
# Integer._eval_power(C 层大整数乘方,运算期间一直持有 GIL),agent loop 的事件循环线程
# 拿不到 GIL,连 asyncio.wait_for 的 30 秒超时都触发不了。线程判分对这种情况无解。
# 这里把判分放进常驻子进程,通过 stdin/stdout 一问一答;父进程这边只是异步读管道,
# 超时就 kill 子进程、补一个新的。子进程卡死只会卡它自己。
_GRADER_SRC = r"""
import sys, os, json
out = os.fdopen(os.dup(1), "w", buffering=1)
os.dup2(2, 1)                         # 库里的 print 都改去 stderr,协议只走 out
sys.path.insert(0, os.environ["CRITICOPD_DIAG_PKG_PARENT"])
os.environ.setdefault("MATH_GRADER_PATH", os.path.join(os.environ["CRITICOPD_DIAG_PKG_PARENT"], "criticopd_diag", "reward", "grader"))
from criticopd_diag.reward.math_reward import compute_score
out.write("@@READY\n")
for line in sys.stdin:
    try:
        q = json.loads(line)
        v = compute_score(solution_str=q["t"], ground_truth=q["g"])
        r = float(v["score"] if isinstance(v, dict) else v) > 0.5
    except Exception:
        r = False
    out.write("@@R " + ("1" if r else "0") + "\n")
"""


class _GraderPool:
    _pools: dict = {}

    def __init__(self, n: int):
        self.n = n
        self.idle: asyncio.Queue = asyncio.Queue()
        self.started = False
        self.lock = asyncio.Lock()

    @classmethod
    def get(cls) -> "_GraderPool":
        loop = asyncio.get_running_loop()
        p = cls._pools.get(id(loop))
        if p is None:
            p = cls._pools[id(loop)] = cls(int(os.environ.get("CRITIC_OPD_GRADER_PROCS", "8")))
        return p

    async def _spawn(self):
        import sys as _sys
        proc = await asyncio.create_subprocess_exec(
            _sys.executable, "-c", _GRADER_SRC, stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL)
        while True:
            line = await asyncio.wait_for(proc.stdout.readline(), timeout=180)
            if not line:
                raise RuntimeError("grader 子进程启动失败")
            if line.startswith(b"@@READY"):
                return proc

    async def _replace(self, proc):
        try:
            proc.kill()
        except ProcessLookupError:
            pass
        asyncio.ensure_future(proc.wait())            # 回收,避免僵尸进程
        for _ in range(5):
            try:
                self.idle.put_nowait(await self._spawn())
                return
            except Exception as e:                    # noqa: BLE001
                logger.warning("[critic_opd] 重建 grader 子进程失败: %s", e)
                await asyncio.sleep(5)

    async def _ensure(self):
        if self.started:
            return
        async with self.lock:
            if self.started:
                return
            for proc in await asyncio.gather(*[self._spawn() for _ in range(self.n)]):
                self.idle.put_nowait(proc)
            self.started = True

    async def grade(self, text: str, gt, timeout: float) -> bool:
        await self._ensure()
        proc = await self.idle.get()
        try:
            proc.stdin.write((json.dumps({"t": text, "g": str(gt)}) + "\n").encode())
            await proc.stdin.drain()

            async def _read():
                while True:
                    line = await proc.stdout.readline()
                    if not line:
                        raise RuntimeError("grader 子进程退出")
                    if line.startswith(b"@@R "):
                        return line[4:5] == b"1"

            r = await asyncio.wait_for(_read(), timeout=timeout)
        except BaseException:
            # 超时 / 取消 / 子进程挂了:这个子进程可能还会吐出一行迟到的结果,
            # 留着它会被下一个请求读走而张冠李戴,所以一律杀掉换新。
            asyncio.ensure_future(self._replace(proc))
            raise
        self.idle.put_nowait(proc)
        return r
