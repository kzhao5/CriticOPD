# 跨模型诊断:在 OLMo 上重做论文图 3、图 4、图 5(a)

论文的图 3、图 4、图 5(a) 都是在同一对模型上测的:学生 Qwen3-1.7B(OPD 第 80 步),teacher Qwen3-4B-Instruct-2507。附录要在 OLMo 上用**完全相同的流程**再测一遍。

本目录是生成论文图的脚本的移植版。实验逻辑、提示词、采样参数和判分器都没有改,只把模型路径和与模型有关的设置(对话模板参数、停止符、长度)移进了配置文件。我们用论文的原始数据核对过:本目录的作图代码算出的数字与论文图逐项相同,见 `reference/`。

论文当前版本全部图的 PDF 在仓库根目录的 `paper_figures/`,对应关系见第九节。

## 一、三张图测什么

| 论文 | 原图 | 内容 | 用到的阶段 |
|---|---|---|---|
| 图 3 | `paper_figures/fig_2_1.pdf` | 从学生回答的各个截断点出发,teacher 与学生续写的成功率 Λ^T、Λ^S。(a) 正确回答;(b) 错误回答,按 teacher 能否从空前缀解出该题分为两组 | `rollouts` → `select` → `cont e1` → `grade e1` |
| 图 4 | `paper_figures/fig_3.pdf` | (a) teacher 续写的结局:答对、落回学生原来的错误答案、其他。(b) teacher 批改截断的解答时,所列错误中真实错误的比例(由 Qwen3-8B 判定),按该截断点的 Λ^T 分组 | 加上 `closing` → `critic` → `judge` |
| 图 5(a) | `paper_figures/fig_4.pdf` 左图 | CriticOPD 断点前后 teacher 与学生的成功率:虚线沿原回答,实线沿学生看着批改写出的新回答 | 加上 `merge_critic`(方法断点)→ `cont e6` → `grade e6` → `cont e7` → `grade e7` |

**样本。**
- 学生对 900 道题各生成一条回答,从中随机选 150 条正确、150 条错误,称为主批次。图 3 和图 4 只用主批次。
- 图 5(a) 另外加入其余全部写完的错误回答,称为扩充批次,与论文附录 "Rewrite Analysis" 一致。Qwen 上扩充批次有 204 条。
- 两个批次都由 `select` 自动选出,不需要另外运行。

## 二、准备

1. **模型。** 三个模型都下载到本地,配置里写本地路径。
   - **学生**:OLMo 主表中 **OPD** 训练后的 checkpoint,取主表所选的那一步(HF 格式)。不要用 base 模型,也不要用 CriticOPD 训练出来的模型。
   - **teacher**:这次 OPD 训练所用的 teacher。学生与 teacher 必须使用同一个分词器;程序会比较两者的词表,不同则直接报错。
   - **判定模型**:`Qwen/Qwen3-8B`,不要换。
2. **环境。** 与仓库根目录相同(`env/`、`setup_env.sh`;我们用的是 vLLM 0.21.0、transformers 5.15.0、torch 2.11.0)。本目录只依赖 vllm、transformers、pyyaml、sympy、numpy 和 matplotlib(见 `requirements.txt`)。计算节点不能联网时,设置 `HF_HUB_OFFLINE=1`。
3. **配置。** 运行 `cp configs/olmo_TEMPLATE.yaml configs/olmo.yaml`,然后填好所有标 `FILL` 的字段。

所有命令都在本目录(`crossmodel_diag/`)下运行。

## 三、需要按 OLMo 修改的地方(只有这些)

| 字段 | 填什么 |
|---|---|
| `student`、`teacher` | 上面两个模型的本地路径 |
| `student_chat_kwargs`、`teacher_chat_kwargs` | 传给 `tokenizer.apply_chat_template` 的额外参数,必须与 OLMo 的 OPD 训练中学生、teacher 的设置一致。Qwen 学生用的是 `{enable_thinking: false}`;如果 OLMo 的模板没有这类开关,就填 `{}` |
| `student_stop_token_ids`、`teacher_stop_token_ids` | 默认 `null`,自动取分词器和 `generation_config.json` 里的 eos。用 `inspect` 打印出来,确认它们就是 OLMo 一轮对话的结束符;不对就手动填 id 列表 |
| `response_budget` | OLMo 的 OPD 训练所用的最大回答长度(Qwen 是 16384)。续写长度等于它减去前缀长度,与训练一致 |
| `max_model_len` | 不小于题目长度加 `response_budget`,且不超过学生和 teacher 的上下文长度(Qwen 用 20480) |
| `critic_max_model_len` | 不超过 teacher 的上下文长度。批改的输入包含完整解答,输出最多 2048 个 token(Qwen 用 34817) |

## 四、不能修改的地方(改了就和论文不可比)

- `data/diag_problems_900.jsonl`:900 道题及其提示词,与论文相同。
- `criticopd_diag/` 下的全部代码,包括:
  - 选样:种子 7,主批次 150 + 150,加扩充批次;
  - 每条回答 7 个截断点(加上空前缀共 8 个);
  - 续写次数 16 / 8 / 8,温度 1.0,top-p 1.0;
  - 批改:批改提示词,贪心解码,最多 2048 个 token,以及与训练相同的截断和泄露处理;
  - 方法断点(论文最终方法):给参考答案,列出全部错误,在最后一个错误处截断,反馈末尾附上正确答案;
  - 判定:Qwen3-8B 思考模式,及其提示词与采样参数;
  - 判分器。
- 配置里标 `keep` 的字段。

如果遇到不改代码就跑不通的情况(例如 OLMo 的模板不支持 system 角色,或上下文不够长),请先告诉我们,不要自行修改。

## 五、运行

**第 1 步:检查(只需 CPU,约一分钟)。**

```bash
python -m criticopd_diag.diag --config configs/olmo.yaml inspect | tee inspect.txt
```

逐项确认:
- `stop ids` 是 OLMo 一轮对话的结束符,与 OPD 训练时 rollout 的停止符相同。
- **student prompt** 与 OPD 训练时学生看到的完全相同,包括对话模板、system 消息和思考开关。
- **teacher prompt** 与 OPD 训练时 teacher 打分所用的模板相同。
- **critic prompt** 中出现了批改的 system 提示词。如果没有,会打印 `WARNING`,请告诉我们。
- **state after the feedback** 中,`[Teacher feedback] ...` 紧接在学生的前缀之后,并且处于同一个 assistant 回合内。
- 任何一行都没有出现 `TOO LONG`。

**第 2 步:冒烟测试(单卡约一小时)。**
1. 把 `configs/olmo.yaml` 里的模型字段、`*_chat_kwargs`、停止符,以及 `response_budget`、`max_model_len`、`critic_max_model_len` 三个长度抄到 `configs/smoke.yaml`。长度必须用真实值:长度太小时大多数回答会写满上限,批改和重写阶段就没有输入。
2. 运行:
   ```bash
   bash scripts/run_all.sh configs/smoke.yaml
   ```
3. 跑通的标准:
   - 最后一行为 `wrote outputs/smoke/return.tar.gz`;
   - `outputs/smoke/summary.log` 中 `[method_cut]` 的断点数大于 0;
   - `outputs/smoke/figs/` 下有 `fig3`、`fig4`、`fig5a` 三张图。

   如果断点数为 0,把 `limit_problems` 改为 200 后重跑。

冒烟测试的样本太少,数字没有意义,只用来确认整条流程能跑通。确认后删除 `outputs/smoke`,不需要发回。

**第 3 步:正式运行。**

- **Slurm(推荐)**:先在 `scripts/slurm_example.sh` 开头改好 `SBATCH_GPU` 和 `SBATCH_CPU`(账户、分区、时长),然后运行:
  ```bash
  bash scripts/slurm_example.sh configs/olmo.yaml
  ```
  各阶段之间用 `afterok` 依赖串起来。最后一个任务 `d_pack` 会生成 `outputs/olmo/return.tar.gz`。
- **单卡顺序运行**:`bash scripts/run_all.sh configs/olmo.yaml`。Qwen 上约需 64 GPU 小时,也就是两天多。
- **中途失败**:
  - 用单卡脚本时,从失败的阶段接着跑,例如:
    ```bash
    bash scripts/run_all.sh configs/olmo.yaml "cont student e6 0 1"
    ```
  - 用 Slurm 时,单独重跑失败的那一片,例如:
    ```bash
    python -m criticopd_diag.diag --config configs/olmo.yaml cont teacher e1 3 12
    ```
    然后重跑它下游的所有阶段。
  - `closing` 会检查 e1 的状态是否齐全,缺了就停下并报告。

**GPU 时间。** 以下是 Qwen 上的参考值(A100 和 H100 混用,单位为 GPU 小时):

| 阶段 | GPU 小时 |
|---|---|
| rollouts | 0.5 |
| e1(主批次) | 22 |
| e1(扩充批次) | 24 |
| critic | 0.6 |
| judge | 1.3 |
| e6 | 4 |
| e7 | 11 |

OLMo 上的耗时取决于回答长度和错误回答的条数:扩充批次 = 写完的错误回答总数 − 150。

## 六、运行中要核对的数字

每个阶段的关键一行都会写进 `outputs/olmo/summary.log`(这个文件会一起打包发回)。下面列出 Qwen 上的对应值:

| 行 | Qwen 上的值 | 核对什么 |
|---|---|---|
| `[rollouts] ... stop ids` | `[151643, 151645]` | 是否为 OLMo 的结束符 |
| `[select]` | 900 条中:正确 517,错误且写完 354,写满上限 29;主批次 150 + 150,扩充批次 204;平均截断点 8.0 | 正确或错误不足 150 条时,样本会相应变少,请告诉我们。写满上限的比例很高,通常说明 `response_budget` 太小或停止符不对 |
| `[closing] main batch` | 150 条错误:A 68,B 19,C 63 | A 类是图 4(b) 和图 5(a) 的样本 |
| `[closing] ext batch` | 204 条错误:A 95,B 31,C 78 | |
| `[critic] ... skipped` | 不出现 | 出现说明 `critic_max_model_len` 不够 |
| `[method_cut]` | 161 个断点,2 个没有可用的错误 | |
| `[judge]` | 各分片之和等于主批次 A 类截断批改中列出了错误的条数 | |

`python -m criticopd_diag.diag --config configs/olmo.yaml check` 会打印各图的样本量和关键数字。可以拿它与 `reference/qwen3_1p7b_numbers.json` 对照字段是否齐全(数值本身预期会不同)。

## 七、需要发回的内容

1. **`outputs/olmo/return.tar.gz`**(几 MB),内含:
   - `figs/{fig3,fig4,fig5a}.{pdf,png}` 和 `figs/numbers.json`;
   - `summary.log`、`config_used.json`;
   - 关闭点、方法断点、批改原文、判定结果,以及每次续写的对错标记(不含续写文本)。

   我们会用它按论文样式重画附录图。
2. **`inspect.txt`**。
3. **模型说明**:
   - 学生的 base 模型名,以及所用 OPD 运行的名字和步数;
   - teacher 的模型名;
   - OPD 训练的最大回答长度;
   - 所用 GPU 型号。

完整的 `outputs/olmo` 有若干 GB(含全部续写文本),不需要发回,但请保留到论文定稿。

## 八、目录

| 位置 | 内容 |
|---|---|
| `criticopd_diag/diag.py` | 各阶段的实现。文件开头列出了全部阶段及其参数 |
| `criticopd_diag/method.py` | 从训练代码 `critic_opd_agent_loop.py` 原样复制的部分:分段、批改提示词、解析、泄露处理、反馈拼接模板、判分进程池 |
| `criticopd_diag/reward/` | 训练所用的判分器 |
| `criticopd_diag/plots.py` | 作图,以及各图数字的定义 |
| `configs/` | `olmo_TEMPLATE.yaml`(要填的模板)、`smoke.yaml`(冒烟测试)、`qwen3_1p7b_reference.yaml`(论文的设置,供对照) |
| `scripts/run_all.sh`、`scripts/slurm_example.sh` | 单卡顺序运行;Slurm 分片运行 |
| `data/diag_problems_900.jsonl` | 900 道题,来自 DAPO-Math-17k |
| `reference/` | 用论文数据重算的 Qwen 图和数字,与论文图一致 |

## 九、`paper_figures/` 的对应关系

| 论文 | 文件 | 标签 |
|---|---|---|
| 图 1 | `schematic.pdf` | `fig:schematic` |
| 图 2 | `tradeoff.pdf` | `fig:tradeoff` |
| 图 3 | `fig_2_1.pdf` | `fig:failure-region` |
| 图 4 | `fig_3.pdf` | `fig:critique` |
| 图 5 | `fig_4.pdf` | `fig:rewrite-effect` |
| 图 6 | `fig_5.pdf` | `fig:example` |
| 图 7 | `fig_hyper.pdf` | `fig:hyper` |
| 图 8 | `fig_rq4.pdf` | `fig:why` |
| 附录 | `fig_cost.pdf`、`fig_dynamics.pdf`、`fig_budget.pdf` | `fig:cost`、`fig:dynamics`、`fig:budget` |
