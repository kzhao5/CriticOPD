# CriticOPD:代码、配置与结果(交接版)

本仓库是论文 Qwen3 主表实验**实际使用的那一版代码**,原样上传,没有事后整理。用途:在新的模型家族(Gemma 3)上用完全相同的训练和评测设置复现主表(全部 baseline + CriticOPD,多 seed)。跨模型只跑主表,不做消融和调参。

- 训练代码基于上游 [zju-real/Relay-OPD](https://github.com/zju-real/Relay-OPD) 的提交 `eab2145`(2026-07-29),其中 `relay-opd/` 是改过的 verl。上游许可证见 `relay-opd/LICENSE`(Apache-2.0)。
- 我们对上游的全部改动:`docs/upstream_diff.patch`(改了 19 个文件),另新增 3 个文件:`relay-opd/verl/experimental/agent_loop/critic_opd_agent_loop.py`(CriticOPD 主体)、`relay-opd/verl/utils/semantic_eos.py`、`relay-opd/opd/patches/tb_steer_traj.py`。
- 所有脚本里的路径都是我们集群上的绝对路径,需要按「七、需要替换的路径」改成你的。

> **CriticOPD 最终版(2026-10-04 确定,跨模型请用这一版):** 批改老师列出**全部**错误(至多 5 个),学生在**最后一个**错误处断开、看着依次列出的全部反馈续写。
> 启动:`ARM=R4GTKL2K bash criticopd/submit_arm.sh`,或 `ONLY=critic bash multiseed/submit.sh`(多 seed,训完自动评测)。
> 与早期的 R4GT(只找第一个错误)相比只多三处:多错误提示词 `CRITIC_SYS_MULTI`(`CRITIC_OPD_KERR=last`);反馈里写出标准答案的错误先去掉(`drop_leaky`);
> 批改输出上限 2048(`CRITIC_OPD_CRITIC_TOKENS=2048`),仍被截断时丢掉最后那条没写完的错误(`drop_incomplete_tail`)。
> 我们的 Qwen3-1.7B 结果:R4GT@40 六项 52.35;全部错误(上限 1024)@40 52.99;最终版正在跑 3 个 seed。

## 目录

| 位置 | 内容 |
|---|---|
| `relay-opd/` | 训练与评测代码(改过的 verl + `opd/` 下的方法脚本、判分、评测) |
| `criticopd/` | CriticOPD 的提交脚本(`submit_arm.sh`)、统一评测提交(`submit_unified.sh`)、分片汇总(`agg_step.sh`、`agg_shards.py`)、离线分析脚本及其报告 |
| `paper_baselines/` | SFT / KD / TRD / GRPO / SKD 的分段训练启动(`train_segment.sl`)与离线造数据(`gen_data.sl`) |
| `multiseed/` | 多 seed 提交脚本(`submit.sh`)、评测守护(`eval_guard.sh`,自动评测、核对题数、补交失败分片) |
| `run_method.sl`、`run_method_freegpu.sl`、`lucid_v2/run_locked.sl` | Slurm 启动器(`lucid_v2/` 只是历史目录名,里面只有这个通用启动器) |
| `eval_shard.sl`、`eval_ood.sl`、`submit_ood.sh`、`eval_mc.sl`、`score_code.sh` | 数学评测分片、OOD 评测(选择题与代码) |
| `configs/<run>/` | 每次训练实际执行的命令(`launch_command.txt`,含全部 Hydra 覆盖)、训练器打印的完整配置(`resolved_config.txt`)、每段 Slurm 任务的节点/GPU/耗时/提交命令(`slurm_jobs.tsv`) |
| `configs/hyperparameters.md` | 各运行关键超参数对照表(从启动命令自动解析) |
| `data/` | 训练集 `dapo-math-17k.parquet`、全部评测集 `bench/`、离线造数据的参数摘要 |
| `reference/` | 用来核对复现的参照:每步训练指标 `metrics/<run>.csv`、原始训练日志 `training_logs/`、69 个 checkpoint 的评测汇总 `eval_summaries/`、CriticOPD@40 的逐条生成 `eval_outputs/`、40 条 critic 输入输出样例 `critic_samples.jsonl` |
| `results_final/` | 论文用的结果表(主表、OOD、消融)、算力统计与图 |
| `env/` | `pip_freeze.txt`、`relay-opd.conda.yml` |

## 一、主表的方法、启动方式与所选 checkpoint

学生 Qwen3-1.7B / Qwen3-0.6B(后训练版,非思考模板),teacher Qwen3-4B-Instruct-2507。所有方法从同一个学生起点训练,batch 128 题,学习率 1e-6 恒定(无 warmup),回答上限 16384(FastOPD 为 8192),采样温度 1.0、top-p 1.0,`data.seed=42`,每 20 步存一次。逐项数值见 `configs/hyperparameters.md`,原始命令见 `configs/<run>/launch_command.txt`。

| 方法 | 训练入口(`METHOD_SCRIPT`) | 损失 | 1.7B 运行 / 所选步 | 0.6B 运行 / 所选步 |
|---|---|---|---|---|
| SFT | `opd/scripts/baselines/sft.sh` | 交叉熵(teacher 离线解答) | `sft_pt17b` @40 | 合作者提供 @139 |
| KD(SeqKD) | `opd/scripts/baselines/seqkd.sh` | top-128 前向 KL(teacher 离线解答) | `seqkd_pt17b` @139 | 合作者提供 @139 |
| GRPO | `opd/scripts/baselines/grpo.sh` | GRPO,每题 8 条,无 KL | `grpo_pt17b` @120 | `grpo_pt06b` @80 |
| OPD | `opd/scripts/baselines/opd.sh` | k1 逆 KL 估计 + 策略梯度 | `opd_1p7b` @80 | `opd_pt06b` @40 |
| TRD | `opd/scripts/baselines/trd.sh` | top-128 前向 KL(teacher 改写学生解答) | `trd_pt17b` @40 | 合作者提供 @139 |
| FastOPD | `opd/scripts/baselines/fastopd/8192.sh` | 同 OPD,回答上限 8192 | `fastopd8192_1p7b` @60 | `fastopd8192_pt06b` @40 |
| SKD | `opd/scripts/baselines/skd.sh` | 学生起草、teacher top-5 接受,top-128 前向 KL | `skd_pt17b` @120 | `skd_pt06b` @40 |
| RelayOPD | `opd/scripts/relay_opd/train.sh` | `relay_opd` | `relay_1p7b` @40 | `relay_pt06b` @60 |
| **CriticOPD** | `opd/scripts/baselines/opd.sh` + `CRITIC_OPD_*` 开关(见二) | k1 + 策略梯度,修复段由带反馈的 teacher 打分(L_fb) | 最终版 `criticopd_final_pt17b_seed{42,43,44}` @40(进行中);上限 1024 的同一算法 `criticopd_R4GTKL_pt17b` @40 为 52.99 | `criticopd_final_pt06b_seed{42,43,44}`,训到 60,评测 40 和 60(进行中) |

**所选步数**:每个方法在训练中评测过的 checkpoint 里,取数学前四项(AIME24、AIME25、AMC23、MATH500)平均最高的一个。所有方法都是恒定学习率,新 seed 只需训到这一步即可,与训到更远再回看完全等价。

**怎么启动**(我们集群上的原始用法):

```bash
# CriticOPD(1.7B,4 张卡:actor 2 + teacher 2,训到第 40 步)
PART=cs CHUNK_END=40 TEST_FREQ=-1 ARM=R4GT bash criticopd/submit_arm.sh
#   调用链:submit_arm.sh -> lucid_v2/run_locked.sl -> run_method_freegpu.sl -> relay-opd/opd/scripts/baselines/opd.sh

# OPD / FastOPD / RelayOPD(seed 42 的 1.7B 用的是 8 卡 run_method.sl;0.6B 与多 seed 用 4 卡 run_method_freegpu.sl)
sbatch --gres=gpu:4 --export=ALL,METHOD_SCRIPT=opd/scripts/baselines/opd.sh,OUTPUT_DIR=...,EXP_ID=...,\
ACTOR_GPUS_PER_NODE=2,TEACHER_GPUS_PER_NODE=2,SAVE_FREQ=20,TEST_FREQ=-1,VAL_BEFORE_TRAIN=False,\
EXTRA_ARGS="trainer.resume_mode=auto trainer.total_training_steps=80" run_method_freegpu.sl

# 全部 baseline 的多 seed(seed 43、44;seed 42 为已有结果),训练完自动评测并核对
bash multiseed/submit.sh               # 1.7B
SIZE=0.6B bash multiseed/submit.sh     # 0.6B
```

`multiseed/submit.sh` 里的 `SPEC17` / `SPEC06` 就是每个方法的脚本、所选步数、训练数据的完整对照,建议直接以它为模板。几个容易漏的环境变量:GRPO 读 `N_GPUS`(默认 8,我们是 4);SFT 读 `NUM_GPUS`;其余读 `ACTOR_GPUS_PER_NODE` / `TEACHER_GPUS_PER_NODE`;随机种子:`SEED`(写入 `data.seed`,默认 42)、`SFT_DATA_SEED`(SFT 的数据打乱,默认 0)。

## 二、CriticOPD 的实现位置

全部在 `relay-opd/verl/experimental/agent_loop/critic_opd_agent_loop.py`(行号以本仓库为准),由环境变量开关控制。最终版对应 `criticopd/submit_arm.sh` 里的 `ARM=R4GTKL2K`:

```
CRITIC_OPD_ENABLE=1 CRITIC_OPD_USE_FEEDBACK=1 CRITIC_OPD_SPLICE=after CRITIC_OPD_LOSS=fb CRITIC_OPD_GIVE_GT=1
CRITIC_OPD_KERR=last CRITIC_OPD_CRITIC_TOKENS=2048
CRITIC_OPD_TRUNC=0 CRITIC_OPD_KEEP_CORRECT=0 CRITIC_OPD_ANNEAL= CRITIC_OPD_REF_FILE=   ROLLOUT_N=1
并在 Hydra 里设 actor_rollout_ref.rollout.agent.default_agent_loop=critic_opd_agent
```

| 环节 | 位置 |
|---|---|
| 哪些 rollout 送去 critic | `run()` 第 626–630 行:写完的用标准答案判分,答错才送;被截断(写满 16384)的在 `GIVE_GT=1` 时也送,critic 认为路线没问题会回 `NONE` |
| 解答切段 | `segment()`(按空行切段) |
| critic 提示词 | `_MULTI_RULES` / `CRITIC_SYS_MULTI` / `CRITIC_SYS_MULTI_CAPPED`(第 190–220 行);用户消息在第 668 行附近(题目 + 标准答案 + 编号后的各段) |
| critic 输出上限与截断 | 第 686–693 行:上限 `max(1024, CRITIC_OPD_CRITIC_TOKENS)` = 2048;撞到上限时 `drop_incomplete_tail()`(第 246 行)去掉最后那块没写完的错误 |
| 输出解析(ERROR / SEGMENT / QUOTE / FEEDBACK / NONE) | `parse_critic_multi()` 第 222 行:按 `SEGMENT:` 分块,每块按引文在原文中定位,定位不到才退回段号;去重、按段号排序,至多 5 个;`NONE` 返回空 |
| 去掉泄露答案的错误 | 第 699 行 `drop_leaky()`(第 261–278 行):反馈里出现标准答案(独立成词)而学生原文里没有的,整条去掉 |
| 选断点、拼反馈 | 第 703 行 `kerr_select(errs, "last")` 取全部错误,断点为最后一个;`kerr_feedback()`(第 279 行)把各条反馈编号拼成一段 |
| 保留前缀、插入反馈 | 第 714 行起:保留到最后一个错误所在段(含该段),按 token 对齐截断;在前缀后接 `FEEDBACK_TMPL` 让学生续写,续写预算 = 16384 − 前缀长度 |
| 训练序列去掉反馈 | 第 740–741 行:训练序列 = 保留前缀(mask 0)+ 学生续写(mask 1),反馈文字不进序列,只对续写算损失 |
| teacher 带反馈打分(L_fb) | 第 775 行把「题目 + 前缀 + 反馈」交出;`relay-opd/verl/experimental/agent_loop/agent_loop.py` 第 1033 行起用它让 teacher 重新给续写 token 打分,覆盖训练序列里续写那几行的 teacher logprob |
| 训练指标 | `relay-opd/verl/trainer/ppo/ray_trainer.py` 第 2107 行起(`critic_opd/repair_seq_frac`、`repair_tok_frac`);agent loop 的计数里 `kerr_k*`(实际用到几个错误)、`kerr_leak_drop`、`kerr_trunc_drop` |

critic 就是 teacher 本身(Qwen3-4B-Instruct-2507),贪心解码(温度 0)。只找第一个错误的旧版(R4GT)用 `CRITIC_SYS_GT` 和 `parse_critic()`,输出上限 512。其余开关(`REF_FILE`、`TRUNC`、`KEEP_CORRECT`、`ANNEAL` 等)是消融,默认关闭。

## 三、数据

- 训练集:`data/dapo-math-17k.parquet`(17917 题;1 个 epoch = 139 步)。提示词就在 `prompt` 列里:`Solve the following math problem step by step. The last line of your response should be of the form Answer: $Answer ... Remember to put your answer on its own line after "Answer:".`,标准答案在 `reward_model.ground_truth`。
- SFT 与 KD 的数据:teacher 对全部 17917 题各生成 1 条解答,温度 1.0、top-p 1.0、上限 16384,用学生的对话模板(非思考)。启动:`MODE=teacher bash paper_baselines/gen_data.sl` → `relay-opd/opd/data/generate_teacher_trajectories.py`。参数摘要见 `data/offline_data_summaries/teacher_traj_pt17b/`(平均回答 4219 token)。Qwen3-0.6B 与 1.7B 的分词器和模板完全相同,0.6B 直接复用同一份。
- TRD 的数据:学生先做(平均 1415 token),teacher 看着学生的解答改写(平均 3173 token)。`MODE=trd bash paper_baselines/gen_data.sl` → `generate_trd_trajectories.py`,与学生相关,换学生要重新生成。
- GRPO 的奖励与所有判分:`relay-opd/opd/reward/math_reward.py` 的 `compute_score`,依次尝试 `\boxed{}` 抽取、`The answer is:` 抽取、math_verify 比较,答对 1 否则 0。CriticOPD 判断「写完是否答对」用的也是它(放在子进程里跑,避免 sympy 卡死)。

## 四、评测

数学(主表):

```bash
RUN=<checkpoint 目录名> STEP=<步数> TAG=<任务名前缀> bash criticopd/submit_unified.sh   # 按题目分片提交
bash criticopd/agg_step.sh <RUN> <STEP>                                               # 分片汇总
```

| benchmark | 文件 | 题数 | 每题采样 |
|---|---|---|---|
| AIME 2024 | `data/bench/aime-24.parquet` | 30 | 32 |
| AIME 2025 | `data/bench/aime-2025.parquet` | 30 | 32 |
| AMC 2023 | `data/bench/amc23.parquet` | 83 | 32 |
| MATH500 | `data/bench/math500.parquet` | 500 | 8 |
| OlympiadBench | `data/bench/olympiadBench.parquet` | 675 | 8 |
| Minerva Math | `data/bench/minerva.parquet` | 272 | 8 |

采样:温度 1.0、top-p 1.0、生成上限 32768(`max_model_len` 34817)、seed 42,非思考模板;指标为 avg@k。评测代码 `relay-opd/opd/eval/math_benchmarks.py`,由 `relay-opd/opd/scripts/evaluation/math.sh` 调用,判分同上。分片合并后的 jsonl 里 `problem_idx` 是分片内的局部编号,做逐题配对时用 `problem_idx × 分片数 + 分片号`。`multiseed/eval_guard.sh` 会自动完成评测、汇总、核对、补交失败分片:每个 benchmark 的所有分片都在才合并(缺一片 `agg_shards.py` 拒绝合并),合并后核对题数、每题采样数,并数一遍逐条记录必须正好是「题数 × 采样数」;缺片只补交缺的那几片(第二轮起推迟 10 分钟,最多 20 轮)。评测分片会同时向多组分区排队,`eval_shard.sl` 里有锁,先开始的副本取消其余,同一分片不会重复跑。训练链见 `multiseed/submit.sh`;训完后 `train_segment.sl` 会取消多备的分段,`multiseed/reap.sh` 可手动清理。

OOD(`submit_ood.sh` → `eval_ood.sl`;选择题 `eval_mc.sl`,代码 `score_code.sh`):每题 4 次采样,温度 1.0;选择题上限 8192,按 `\boxed{字母}` 判分;代码上限 4096,EvalPlus pass@1(base 与 plus 测试,每次运行前清空 EvalPlus 结果缓存)。数据:`mmlu_mini`(MMLU 按学科分层抽 2000/14042)、`mmlu_pro_2k`(MMLU-Pro 按类别分层抽 2000/12032)、`arc_c`(1172)、`obqa`(500)、`humanevalplus`(164)、`mbpp`(MBPP+,378)。完整协议写在 `results_final/ood_table.json` 的 `protocol` 字段里。

## 五、环境

torch 2.11.0(CUDA 13.0)、vLLM 0.21.0、transformers 5.15.0、ray 2.57.0、sympy 1.14.0、numpy 1.26.4;没有安装 flash-attn。完整列表:`env/pip_freeze.txt`、`env/relay-opd.conda.yml`;安装脚本 `setup_env.sh`。

我们集群上必需的设置(都已写在启动器里):`OPENSSL_CONF=/dev/null`(FIPS);计算节点没有 nvcc,所以 `VLLM_USE_FLASHINFER_SAMPLER=0`;H100/H200/B200(sm90 及以上)上自带的 NCCL 2.28.9 会段错误,`run_method_freegpu.sl` 会预加载 NCCL 2.29.7(`pip install --target criticopd/nccl_alt/2.29.7 nvidia-nccl-cu12==2.29.7`,二进制太大没有上传);B200 节点需另载 CUDA 12.8 模块。

## 六、核对复现用的参照

- `results_final/main_table_math.json`、`ood_table.json`、`criticopd_ablations.json`:论文里的数字。注意主表已改为统一协议(上表),以 `reference/eval_summaries/<run>_u32/step_<N>/` 为准,例如 CriticOPD@40:AIME24 38.23、AIME25 26.35、AMC23 66.68、MATH500 87.98、OlympiadBench 57.94、Minerva 36.90,六项平均 52.35;OPD@80 为 50.18。
- `reference/metrics/<run>.csv`:每一步的全部训练指标(回答长度、loss、teacher/学生耗时、CriticOPD 的修复比例等),相当于 wandb 导出。
- `reference/eval_outputs/criticopd_R4GT_pt17b_step40/`:CriticOPD@40 六个 benchmark 的逐条生成文本。
- `reference/critic_samples.jsonl`:40 条 critic 输入(题目、标准答案、学生解答、编号后的用户消息)、critic 原始输出、解析结果,以及每条对应的两次修复和对错。
- 算力:`results_final/step_time.csv` 与 `fig_step_time.*`(4×A100 上每一步的 GPU 时间),`configs/<run>/slurm_jobs.tsv` 有每段任务的实际耗时和 GPU 型号。

## 七、需要替换的路径

| 我们的路径 | 含义 |
|---|---|
| `/home/kzhao2/Relay-OPD` | 本仓库根目录 |
| `/home/kzhao2/OPD/model/Qwen3-1.7B`、`Qwen3-0.6B`、`Qwen3-4B-Instruct-2507` | 学生与 teacher 模型 |
| `/home/kzhao2/OPD/datasets/dapo-math-17k.parquet` | 训练集(本仓库 `data/`) |
| `/home/kzhao2/Relay-OPD/outputs/data/bench` | 评测集(本仓库 `data/bench/`) |
| `/home/kzhao2/Relay-OPD/outputs/{checkpoints,eval_out,offline_data}` | 输出目录 |
| `/home/kzhao2/.conda/envs/relay-opd` | conda 环境 |
| `--account=als44`、分区 `cs`/`cs2`/`cs3`/`dw`/`m13h`、QOS `cs`/`dw87`/`gpu` | Slurm 设置 |

## 八、换成 Gemma 3 时要改的地方

- **停止符**:Qwen 的 `151643` / `151645` 写死在 `paper_baselines/gen_data.sl`(`GEN_STOP_TOKEN_IDS`)、`relay-opd/opd/data/generate_teacher_trajectories.py`、`generate_trd_trajectories.py`、`submit_ood.sh`、`criticopd/` 的离线分析脚本里。训练时 rollout 用的是模型自带的 EOS,需要确认 Gemma 的 `generation_config.json` 包含 `<end_of_turn>`(id 106)。
- **critic 的 system 角色**:`critic_opd_agent_loop.py` 构造 critic 消息时用 `{"role": "system"}` 传提示词(最终版在第 668 行)。Gemma 3 的模板会把 system 合并进第一条 user 消息,critic 实际看到的输入格式会变,需要先确认。
- **思考模式开关**:`enable_thinking` / `disable_thinking` 是 Qwen 模板专用参数,出现在 `opd/eval/math_benchmarks.py`、`opd/scripts/evaluation/math.sh`、各训练脚本和造数据脚本里,换模型时去掉或忽略。
- Gemma 3 的 4B 以上是多模态模型类,训练和 vLLM 里需要按纯文本加载。

## 九、我们踩过的坑

- verl 的数据顺序取决于 `data.dataloader_num_workers`:同一个 seed,8 个进程与单进程的顺序不同(都是确定的)。
- 判分要放进子进程:sympy 遇到 `\boxed{10^{10^9}}` 这类答案会占住 GIL,线程超时也拦不住。
- 我们集群上有节点会在作业启动 2 秒内由 root 杀掉作业(健康检查),`multiseed/eval_guard.sh` 会自动重交;这是集群问题,与代码无关。
