# CriticOPD: 学生写完整 -> 判错 -> teacher 指出首处错误并给反馈 -> 学生带反馈重写。
#
# 和 Relay/Lucid 的区别:那两个是在采样过程中"换手"(teacher 从同一前缀续写),这里是
# 事后批评 + 重写,teacher 在一个完全不同的 prompt(critic prompt)上生成,不参与解题。
#
# 训练侧的约定(response_mask):
#   保留的前缀  -> 0   反馈 -> 不进序列   学生重写的 token -> 1
# 也就是说,反馈只在生成时出现在上下文里,训练时被去掉,损失只落在学生自己写的修复 token 上。
# 这正是"让学生把修正内化"的意思:测试时没有 teacher,它必须自己写出修正后的那段。
import asyncio
import json
import random
import logging, os, re
from typing import Any
from uuid import uuid4

from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopOutput, register
from verl.utils.profiler import simple_timer
from verl.utils.rollout_trace import rollout_trace_op
from verl.workers.rollout.replica import TokenOutput

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
sys.path.insert(0, "/home/kzhao2/Relay-OPD/relay-opd")
os.environ.setdefault("MATH_GRADER_PATH", "/home/kzhao2/Relay-OPD/relay-opd/opd/reward/grader")
from opd.reward.math_reward import compute_score
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


@register("critic_opd_agent")
class CriticOpdAgentLoop(AgentLoopBase):
    # 新开关的类级默认值:任何绕过 __init__ 的构造路径下也有定义。否则 _run 里读到不存在的
    # 属性会抛 AttributeError,被 run() 外层的 try/except 静默降级成普通 rollout ——
    # 和之前 asyncio 漏 import 是同一类静默失效。
    give_gt = False
    trunc = False
    trunc_max = 512
    keep_correct = False
    ref_file = None
    critic_maxlen = 34817
    kerr = None
    prefix_loss = False
    critic_all = False
    _REFS = None            # {sha1(题目文本): 参考解答},每个进程加载一次
    anneal = None
    global_step = -1

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prompt_length = self.rollout_config.prompt_length
        self.response_length = self.rollout_config.response_length
        # SPLICE=before 对应"选项 1"(不保留出错段),=after 对应"选项 2"(保留出错段让学生反思)
        self.splice = os.environ.get("CRITIC_OPD_SPLICE", "before").strip().lower()
        self.enabled = os.environ.get("CRITIC_OPD_ENABLE", "1") == "1"
        self.use_feedback = os.environ.get("CRITIC_OPD_USE_FEEDBACK", "1") == "1"  # 0 = R2 的无反馈对照
        self.critic_max_tokens = int(os.environ.get("CRITIC_OPD_CRITIC_TOKENS", "512"))
        self.give_gt = os.environ.get("CRITIC_OPD_GIVE_GT", "0") == "1"
        self.trunc = os.environ.get("CRITIC_OPD_TRUNC", "0") == "1"
        self.trunc_max = int(os.environ.get("CRITIC_OPD_TRUNC_MAX", "512"))
        # 只保留修对的重写:重写后用 gt 再判一次,没修对就退回原始 rollout 按 plain OPD 训练。
        self.keep_correct = os.environ.get("CRITIC_OPD_KEEP_CORRECT", "0") == "1"
        self.ref_file = os.environ.get("CRITIC_OPD_REF_FILE") or None
        self.critic_maxlen = int(os.environ.get("CRITIC_OPD_CRITIC_MAXLEN", "34817"))   # teacher 的 max_model_len
        # k 消融:'1'/'2'/'3'/'last';不设则与原来完全一样。只在「保留出错段」下有意义。
        self.kerr = (os.environ.get("CRITIC_OPD_KERR", "").strip().lower() or None)
        # 前缀算损失:第一个错误所在段之前的学生 token 也按普通 OPD 训练(teacher 不看反馈);不设则与原来一样全部 mask
        self.prefix_loss = os.environ.get("CRITIC_OPD_PREFIX_LOSS", "0") == "1"
        # 消融:去掉判分这道门,答对的 rollout 也交给 critic(用中性提示词,没有真正的错误就回 NONE、按普通 OPD 训练)
        self.critic_all = os.environ.get("CRITIC_OPD_ALL", "0") == "1"
        # 退火(DAgger 式 beta 调度):"开始衰减步,衰减结束步,beta 下限"。答错的样本以概率 beta
        # 走 critic 修复,否则保留它自己的 rollout 按 plain OPD 训练。不设则恒为 1(即 R4GT)。
        _a = os.environ.get("CRITIC_OPD_ANNEAL", "").strip()
        self.anneal = tuple(float(x) for x in _a.split(",")) if _a else None
        self.global_step = int(kwargs.get("critic_global_step", -1) or -1)
        self._grader = None

    @staticmethod
    def _disarm_sigalrm():
        """判分器只能在主线程里安全使用 SIGALRM;agent loop 的事件循环跑在非主线程。

        openmathinst_utils.time_limit 的顺序是
            signal.setitimer(ITIMER_REAL, s)   # 进程级,任何线程都成功
            signal.signal(SIGALRM, handler)    # 非主线程抛 ValueError
        异常发生在 yield 之前,finally 里的"解除定时器"永远不执行 -> 10 秒后 SIGALRM
        以默认动作(终止进程)送达。没有 traceback、没有 core,raylet 只看到
        "Worker unexpectedly exits with a connection error code 2"。
        math_verify 的源码里也写明了同一件事,并建议关掉它的 timeout。

        这里把 SIGALRM 这一条链整体变成空操作:定时器不再真的 arm(所以永远不会有
        信号送达),而 signal.signal(SIGALRM, ...) 不再抛异常(所以 finally 正常走完,
        判分逻辑和离线完全一致)。其他信号不碰,Ray 自己的 SIGINT/SIGTERM 处理不受影响。
        """
        import signal as _sg
        if getattr(_sg, "_critic_opd_disarmed", False):
            return
        _orig = _sg.signal

        def _sig(sig, handler):
            if sig == _sg.SIGALRM:
                return _sg.SIG_DFL
            return _orig(sig, handler)

        _sg.signal = _sig
        _sg.setitimer = lambda *a, **k: (0.0, 0.0)
        _sg.alarm = lambda *a, **k: 0
        _sg._critic_opd_disarmed = True

    def _grade(self, text, gt):
        if self._grader is None:
            import sys
            sys.path.insert(0, "/home/kzhao2/Relay-OPD/relay-opd")
            os.environ.setdefault("MATH_GRADER_PATH",
                                  "/home/kzhao2/Relay-OPD/relay-opd/opd/reward/grader")
            self._disarm_sigalrm()
            from opd.reward.math_reward import compute_score
            self._grader = compute_score
        try:
            v = self._grader(solution_str=text, ground_truth=gt)
            return float(v["score"] if isinstance(v, dict) else v) > 0.5
        except Exception:
            return False

    def _reference(self, question: str):
        if not self.ref_file:
            return None
        if type(self)._REFS is None:
            import hashlib as _hl, json as _js
            type(self)._REFS = _js.load(open(self.ref_file))
            type(self)._sha1 = staticmethod(lambda t: _hl.sha1(t.encode()).hexdigest())
            logger.warning("[critic_opd] 已加载参考解答 %d 条: %s", len(type(self)._REFS), self.ref_file)
        return type(self)._REFS.get(type(self)._sha1(question))

    def _beta(self) -> float:
        if not self.anneal or self.global_step < 0:
            return 1.0
        t0, t1, bmin = self.anneal
        if self.global_step <= t0:
            return 1.0
        if self.global_step >= t1:
            return bmin
        return 1.0 - (1.0 - bmin) * (self.global_step - t0) / max(t1 - t0, 1e-9)

    async def _verify_async(self, text, gt) -> bool:
        """校验重写是否修对。超时 / 出错按「没修对」处理 —— 保守方向是不把可疑的重写放进训练。"""
        try:
            return bool(await asyncio.wait_for(_GraderPool.get().grade(text, gt, 30), timeout=240))
        except (TimeoutError, asyncio.TimeoutError, RuntimeError, OSError) as e:
            logger.warning("[critic_opd] 校验重写失败(%s),按没修对处理", type(e).__name__)
            return False

    async def _grade_async(self, text, gt):
        """决定要不要触发 critic。超时 / 出错按「答对」处理(不触发 critic),是保守方向。"""
        try:
            return bool(await asyncio.wait_for(_GraderPool.get().grade(text, gt, 30), timeout=240))
        except (TimeoutError, asyncio.TimeoutError, RuntimeError, OSError) as e:
            logger.warning("[critic_opd] 判分失败(%s),按答对处理(不触发 critic)", type(e).__name__)
            return True
    async def _teacher_generate(self, prompt_ids, max_tokens):
        """让 teacher 在给定 prompt 上自由生成。teacher client 原本只用于打分
        (max_tokens=1 + prompt_logprobs),换一组 sampling_params 就是自由生成。"""
        tm = self.teacher_manager
        if tm is None: return None
        key = next(iter(tm.teacher_model_configs))
        out = await tm.teacher_client[key].generate(
            request_id=uuid4().hex, prompt_ids=prompt_ids,
            sampling_params={"max_tokens": max_tokens, "temperature": 0.0},
        )
        return out.token_ids

    _CNT = {"n": 0, "capped": 0, "correct": 0, "attempted": 0, "parsed": 0,
            "repaired": 0, "exc": 0}

    def _tally(self, **kw):
        c = type(self)._CNT
        c["n"] += 1
        for k in kw:
            c[k] = c.get(k, 0) + 1
        if c["n"] % 64 == 0:
            logger.warning("[critic_opd] 样本=%d 截断=%d 答对=%d 尝试=%d 解析成功=%d 修复=%d 异常=%d",
                           c["n"], c["capped"], c["correct"], c["attempted"],
                           c["parsed"], c["repaired"], c["exc"])
            print(f"[critic_opd] tally {c}", flush=True)

    @rollout_trace_op
    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        try:
            return await self._run(sampling_params, **kwargs)
        except Exception:
            # Ray actor 猝死时异常不会冒泡,主进程只看到 ActorDiedError。先打出来再决定。
            import traceback
            self._tally(exc=1)
            logger.error("[critic_opd] run() 异常,降级为普通 rollout:\n%s", traceback.format_exc())
            print("[critic_opd] EXCEPTION\n" + traceback.format_exc(), flush=True)
            return await self._plain(sampling_params, **kwargs)

    async def _plain(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        """出错时的安全退路:就按 single_turn 的方式跑一次,不做 critic/修复。"""
        messages = list(kwargs["raw_prompt"])
        mm = await self.process_multi_modal_info(messages)
        mmk = self._get_mm_processor_kwargs(mm.get("audios"))
        prompt_ids = await self.apply_chat_template(
            messages, images=mm.get("images"), videos=mm.get("videos"),
            audios=mm.get("audios"), mm_processor_kwargs=mmk)
        out: TokenOutput = await self.server_manager.generate(
            request_id=uuid4().hex, prompt_ids=prompt_ids,
            sampling_params=sampling_params, mm_processor_kwargs=mmk)
        rlen = int(sampling_params.get("max_tokens") or self.response_length)
        resp = out.token_ids[:rlen]
        # 降级路径也必须带 response_logprobs:_postprocess 会对整批做
        #   torch.cat([input.response_logprobs for input in inputs])
        # 同批次里只要有一个样本是 None 就会抛
        #   "TypeError: expected Tensor as element N in argument 0, but got NoneType"。
        # R3v2 (13940177) 就是在第 80 步撞上这个,整个 job 以 rc=1 结束
        # (checkpoint 已先行写出,所以没丢数据,但 job 失败了)。
        _lp = list(out.log_probs[:len(resp)]) if out.log_probs is not None else None
        if _lp is not None and len(_lp) < len(resp):
            _lp += [0.0] * (len(resp) - len(_lp))
        return AgentLoopOutput(
            prompt_ids=prompt_ids, response_ids=resp, response_mask=[1] * len(resp),
            response_logprobs=_lp,
            multi_modal_data=mm, mm_processor_kwargs=mmk, num_turns=2,
            metrics={"num_preempted": -1},
            extra_fields={**(out.extra_fields or {}), "critic_opd": {"fallback": 1}})

    async def _run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        messages = list(kwargs["raw_prompt"])
        mm = await self.process_multi_modal_info(messages)
        mmk = self._get_mm_processor_kwargs(mm.get("audios"))
        prompt_ids = await self.apply_chat_template(
            messages, images=mm.get("images"), videos=mm.get("videos"),
            audios=mm.get("audios"), mm_processor_kwargs=mmk)

        metrics = {}
        with simple_timer("generate_sequences", metrics):
            out: TokenOutput = await self.server_manager.generate(
                request_id=uuid4().hex, prompt_ids=prompt_ids,
                sampling_params=sampling_params, mm_processor_kwargs=mmk)
        rlen = int(sampling_params.get("max_tokens") or self.response_length)
        resp = out.token_ids[:rlen]
        stats = {"critic_attempted": 0, "critic_parsed": 0, "repaired": 0}

        gt = None
        rm = kwargs.get("reward_model")
        if isinstance(rm, dict): gt = rm.get("ground_truth")

        # 写完了:用 gt 精确判分,只有答错才介入(比让 teacher 判对错准确且免费)。
        # 写不完(顶上限):没有答案可判分。原设计直接跳过;GT 思路下交给拿着正确答案的
        # teacher 判断这条路还走不走得通,它觉得没问题会回 NONE,那就不介入。
        capped = len(out.token_ids) >= rlen
        text = self.tokenizer.decode(resp, skip_special_tokens=True)
        ready = self.enabled and self.teacher_manager is not None and gt is not None
        is_correct = False
        if capped:
            need = ready and self.give_gt
        else:
            is_correct = bool(ready) and await self._grade_async(text, gt)
            need = ready and (not is_correct or self.critic_all)
            if is_correct and need:
                stats["critic_on_correct"] = 1
                self._tally(correct_critiqued=1)

        if capped:
            self._tally(capped=1)
        elif not need:
            self._tally(correct=1)
        if need and self.anneal is not None:
            _b = self._beta()
            stats["anneal_beta"] = _b
            if random.random() >= _b:
                need = False
                stats["anneal_skip"] = 1
                self._tally(anneal_skip=1)

        if need:
            self._tally(attempted=1)
            stats["critic_attempted"] = 1
            segs = segment(text)
            body = "\n\n".join(f"[{i}] {s}" for i, s in enumerate(segs))
            ref = self._reference(messages[-1]["content"]) if (self.give_gt and not self.kerr) else None
            cids = None
            if ref is not None:
                cmsg = [{"role": "system", "content": CRITIC_SYS_REF_CAPPED if capped else CRITIC_SYS_REF},
                        {"role": "user", "content":
                         f"Problem:\n{messages[-1]['content']}\n\n"
                         f"Correct answer (reference only -- never reveal it): {gt}\n\n"
                         "Reference solution (correct; the student may use a different "
                         f"valid approach):\n{ref}\n\n"
                         f"Student's solution, split into numbered segments:\n\n{body}"}]
                cids = self.tokenizer.encode(self.tokenizer.apply_chat_template(
                    cmsg, add_generation_prompt=True, tokenize=False), add_special_tokens=False)
                if len(cids) + self.critic_max_tokens > self.critic_maxlen:
                    cids = None                     # 加上参考解答放不下:退回不带参考的批改
                    stats["critic_ref_overflow"] = 1
            stats["critic_ref"] = int(cids is not None)
            if self.ref_file:
                self._tally(**({"ref_used": 1} if cids is not None else {"ref_missing": 1}))
            if cids is None and self.kerr:
                cmsg = [{"role": "system", "content": CRITIC_SYS_MULTI_CAPPED if capped else (CRITIC_SYS_MULTI_ANY if is_correct else CRITIC_SYS_MULTI)},
                        {"role": "user", "content":
                         f"Problem:\n{messages[-1]['content']}\n\n"
                         f"Correct answer (reference only -- never reveal it): {gt}\n\n"
                         f"Student's solution, split into numbered segments:\n\n{body}"}]
            elif cids is None and self.give_gt:
                cmsg = [{"role": "system", "content": CRITIC_SYS_GT_CAPPED if capped else CRITIC_SYS_GT},
                        {"role": "user", "content":
                         f"Problem:\n{messages[-1]['content']}\n\n"
                         f"Correct answer (reference only -- never reveal it): {gt}\n\n"
                         f"Student's solution, split into numbered segments:\n\n{body}"}]
            elif cids is None:
                cmsg = [{"role": "system", "content": CRITIC_SYS},
                        {"role": "user", "content":
                         f"Problem:\n{messages[-1]['content']}\n\nStudent's solution, numbered segments:\n\n{body}"}]
            if cids is None:
                cids = self.tokenizer.apply_chat_template(cmsg, add_generation_prompt=True, tokenize=False)
                cids = self.tokenizer.encode(cids, add_special_tokens=False)
            _cmax = max(1024, self.critic_max_tokens) if self.kerr else self.critic_max_tokens
            ctok = await self._teacher_generate(cids, _cmax)
            if ctok:
                ctext = self.tokenizer.decode(ctok, skip_special_tokens=True)
                if self.kerr and len(ctok) >= _cmax:
                    # 多错误批改被截断:最后一条反馈没写完,学生会带着半句话续写 —— 去掉这一块,
                    # 在前一个完整的错误处断开(离线:上限 1024 时 25.6% 被截断,2048 时 5.4%)
                    ctext = drop_incomplete_tail(ctext)
                    stats["critic_truncated"] = 1
                    self._tally(kerr_trunc_drop=1)
                if self.kerr:
                    errs = parse_critic_multi(ctext, segs)
                    n_raw = len(errs)
                    _first_raw = errs[0][0] if errs else None   # 前缀损失的边界:批改找到的第一个错误(含被泄露过滤去掉的)
                    errs = drop_leaky(errs, gt, text)
                    if n_raw > len(errs):
                        stats["critic_n_leak"] = n_raw - len(errs)
                        self._tally(kerr_leak_drop=n_raw - len(errs))
                    sel = kerr_select(errs, self.kerr)
                    idx, fb = (sel[-1][0], kerr_feedback(sel)) if sel else (None, None)
                    first_idx = _first_raw if sel else None
                    stats["critic_n_err"] = len(errs); stats["critic_k_eff"] = len(sel)
                    if sel:
                        self._tally(**{f"kerr_k{len(sel)}": 1})
                else:
                    idx, fb = parse_critic(ctext, segs)
                    first_idx = idx
                if idx is not None and fb:
                    stats["critic_parsed"] = 1
                    self._tally(parsed=1)
                    # k 消融强制保留出错段:不保留时,前 k-1 个错误留在上下文里、第 k 个却被删掉,批语说不清
                    keep = segs[:idx] if (self.splice == "before" and not self.kerr) else segs[:idx + 1]
                    kept_txt = "\n\n".join(keep)
                    _k = _tok_prefix_for_text(self.tokenizer, resp, kept_txt)
                    kept_ids = resp[:_k]
                    # 重写的上下文必须和真正保留下来的 token 完全一致,所以用 kept_ids 解回去,
                    # 而不是用 kept_txt(两者可能差几个空白字符)。
                    kept_txt = self.tokenizer.decode(kept_ids, skip_special_tokens=True)
                    ctx = kept_txt + (FEEDBACK_TMPL.format(fb=fb) if self.use_feedback else "\n\n")
                    ctx_ids = self.tokenizer.encode(ctx, add_special_tokens=False)
                    budget = max(256, rlen - len(kept_ids))
                    if self.trunc:
                        budget = min(budget, self.trunc_max)
                    with simple_timer("repair", metrics):
                        r: TokenOutput = await self.server_manager.generate(
                            request_id=uuid4().hex, prompt_ids=prompt_ids + ctx_ids,
                            sampling_params={**sampling_params, "max_tokens": budget},
                            mm_processor_kwargs=mmk)
                    _rep_ids = list(r.token_ids)
                    _rep_lp = list(r.log_probs) if r.log_probs is not None else None
                    if self.trunc:
                        _kc = _cut_at_blank_line(self.tokenizer, _rep_ids)
                        _rep_ids = _rep_ids[:_kc]
                        if _rep_lp is not None:
                            _rep_lp = _rep_lp[:_kc]
                        stats["repair_tokens"] = _kc
                    # 训练序列 = 保留前缀(mask 0) + 学生重写(mask 1)。反馈不进序列。
                    _resp_r = (kept_ids + _rep_ids)[:rlen]
                    _npre = 0
                    if self.prefix_loss and first_idx:
                        _npre = min(_tok_prefix_for_text(self.tokenizer, resp, "\n\n".join(segs[:first_idx])), len(kept_ids))
                        stats["prefix_loss_tokens"] = _npre
                    _mask_r = ([1] * _npre + [0] * (len(kept_ids) - _npre) + [1] * len(_rep_ids))[:rlen]
                    # 只保留修对的重写:没修对(含顶上限、没写出答案)就不返回修复结果,
                    # 落到函数末尾,按原始 rollout 的 plain OPD 训练(num_turns=2)。
                    _accept = True
                    if self.keep_correct:
                        _accept = await self._verify_async(
                            self.tokenizer.decode(_resp_r, skip_special_tokens=True), gt)
                        stats["repair_verified"] = int(_accept)
                        self._tally(**({"repair_ok": 1} if _accept else {"repair_bad": 1}))
                    if _accept:
                        resp, mask = _resp_r, _mask_r
                        stats["repaired"] = 1
                        type(self)._CNT["repaired"] += 1
                        metrics.setdefault("num_preempted", -1)
                        # 漏了 response_logprobs 时,_postprocess 会在
                        #   torch.cat([input.response_logprobs for input in inputs])
                        # 上撞到 None(未修复的样本有、修复的没有)。前缀用原始 rollout 的
                        # logprobs(_k 是原始 token 数,所以严格对齐),重写段用它自己的。
                        _lp = None
                        if out.log_probs is not None:
                            _rlp = _rep_lp if _rep_lp is not None else [0.0] * len(_rep_ids)
                            _lp = (list(out.log_probs[:_k]) + _rlp)[:len(resp)]
                            if len(_lp) < len(resp):          # 长度必须和 response_ids 一致
                                _lp += [0.0] * (len(resp) - len(_lp))
                        # L_fb(R4):teacher 要在"带反馈"的上下文下给修复 token 重新打分,
                        # 学生则在"无反馈"的序列上被训练 —— 这才是把修正内化的严格形式。
                        # 这里只把重打分所需的材料带出去,真正的 teacher 前向在
                        # AgentLoopWorker._compute_teacher_logprobs_single_sample 里做
                        # (那里本来就持有 teacher_server_manager 和路由键)。
                        # 修复 token 数要按截断后的实际长度算,不能用 len(r.token_ids)。
                        _fb = {}
                        if os.environ.get("CRITIC_OPD_LOSS") == "fb":
                            _nrep = max(0, len(resp) - len(kept_ids))
                            if _nrep > 0:
                                _fb = {"critic_fb_prompt_ids": prompt_ids + ctx_ids,
                                       "critic_fb_n_kept": len(kept_ids),
                                       "critic_fb_n_repair": _nrep}
                        return AgentLoopOutput(
                            prompt_ids=prompt_ids, response_ids=resp, response_mask=mask,
                            response_logprobs=_lp,
                            multi_modal_data=mm, mm_processor_kwargs=mmk, num_turns=4,
                            metrics=metrics, extra_fields={**(r.extra_fields or {}),
                                                           "critic_opd": stats,
                                                           **_fb})

        if metrics.get("num_preempted") is None:
            metrics["num_preempted"] = out.num_preempted if out.num_preempted is not None else -1
        return AgentLoopOutput(
            prompt_ids=prompt_ids, response_ids=resp, response_mask=[1] * len(resp),
            response_logprobs=out.log_probs[:rlen] if out.log_probs is not None else None,
            multi_modal_data=mm, mm_processor_kwargs=mmk, num_turns=2, metrics=metrics,
            extra_fields={**(out.extra_fields or {}), "critic_opd": stats})
