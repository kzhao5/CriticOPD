# 各运行显式传入的关键超参数

自动从 configs/<run>/launch_command.txt 解析;空白表示命令里没有显式覆盖,取 verl 默认值(见同目录 resolved_config.txt)。

| run | batch | mini_batch | lr | max_resp | rollout_n | temp | top_p | loss | topk | actor_gpus | teacher_gpus | save_freq | total_steps | data_seed | num_workers | agent_loop | max_prompt |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| criticopd_R1_pt17b | 128 | 128 | 1e-6 | 16384 | 2 | 1.0 | 1.0 | k1 |  | 2 | 1 | 20 | 80 | 42 |  |  | 2048 |
| criticopd_R2v2_pt17b | 128 | 128 | 1e-6 | 16384 | 1 | 1.0 | 1.0 | k1 |  | 2 | 2 | 20 | 80 | 42 |  | critic_opd_agent | 2048 |
| criticopd_R3v2_pt17b | 128 | 128 | 1e-6 | 16384 | 1 | 1.0 | 1.0 | k1 |  | 2 | 2 | 20 | 80 | 42 |  | critic_opd_agent | 2048 |
| criticopd_R4GTA_pt17b | 128 | 128 | 1e-6 | 16384 | 1 | 1.0 | 1.0 | k1 |  | 2 | 2 | 20 | 60 | 42 |  | critic_opd_agent | 2048 |
| criticopd_R4GTB_pt17b | 128 | 128 | 1e-6 | 16384 | 1 | 1.0 | 1.0 | k1 |  | 2 | 2 | 20 | 40 | 42 |  | critic_opd_agent | 2048 |
| criticopd_R4GTK2_pt17b | 128 | 128 | 1e-6 | 16384 | 1 | 1.0 | 1.0 | k1 |  | 2 | 2 | 20 | 40 | 42 |  | critic_opd_agent | 2048 |
| criticopd_R4GTKL_pt17b | 128 | 128 | 1e-6 | 16384 | 1 | 1.0 | 1.0 | k1 |  | 2 | 2 | 20 | 40 | 42 |  | critic_opd_agent | 2048 |
| criticopd_R4GTR2_pt17b | 128 | 128 | 1e-6 | 16384 | 1 | 1.0 | 1.0 | k1 |  | 2 | 2 | 20 | 40 | 42 |  | critic_opd_agent | 2048 |
| criticopd_R4GTV_pt17b | 128 | 128 | 1e-6 | 16384 | 1 | 1.0 | 1.0 | k1 |  | 2 | 2 | 20 | 80 | 42 |  | critic_opd_agent | 2048 |
| criticopd_R4GT_pt17b | 128 | 128 | 1e-6 | 16384 | 1 | 1.0 | 1.0 | k1 |  | 2 | 2 | 20 | 80 | 42 |  | critic_opd_agent | 2048 |
| criticopd_R4TR_pt17b | 128 | 128 | 1e-6 | 16384 | 1 | 1.0 | 1.0 | k1 |  | 2 | 2 | 20 | 80 | 42 |  | critic_opd_agent | 2048 |
| criticopd_R4_pt17b | 128 | 128 | 1e-6 | 16384 | 1 | 1.0 | 1.0 | k1 |  | 2 | 2 | 20 | 80 | 42 |  | critic_opd_agent | 2048 |
| fastopd8192_1p7b | 128 | 128 | 1e-6 | 8192 | 1 | 1.0 | 1.0 | k1 |  | 4 | 4 | 10 |  | 42 |  |  | 2048 |
| fastopd8192_pt06b | 128 | 128 | 1e-6 | 8192 | 1 | 1.0 | 1.0 | k1 |  | 2 | 2 | 20 |  | 42 |  |  | 2048 |
| grpo_pt06b | 128 | 128 | 1e-6 | 16384 | 8 | 1.0 | 1.0 |  |  | 4 |  | 20 |  | 42 |  |  | 2048 |
| grpo_pt17b | 128 | 128 | 1e-6 | 16384 | 8 | 1.0 | 1.0 |  |  | 4 |  | 10 |  | 42 |  |  | 2048 |
| opd_1p7b | 128 | 128 | 1e-6 | 16384 | 1 | 1.0 | 1.0 | k1 |  | 4 | 4 | 10 |  | 42 |  |  | 2048 |
| opd_pt06b | 128 | 128 | 1e-6 | 16384 | 1 | 1.0 | 1.0 | k1 |  | 2 | 2 | 20 |  | 42 |  |  | 2048 |
| relay_1p7b | 128 | 128 | 1e-6 | 16384 | 1 | 1.0 | 1.0 | relay_opd |  | 4 | 4 | 10 |  | 42 |  |  | 2048 |
| relay_pt06b | 128 | 128 | 1e-6 | 16384 | 1 | 1.0 | 1.0 | relay_opd |  | 2 | 2 | 20 |  | 42 |  |  | 2048 |
| seqkd_pt17b | 128 | 128 | 1e-6 | 16384 | 1 | 1.0 | 1.0 | forward_kl_topk_teacher_renorm | 128 | 2 | 2 | 20 |  | 42 |  |  | 2048 |
| sft_pt17b | 128 |  | 1e-6 |  |  |  |  |  |  |  |  | 20 |  |  |  |  |  |
| skd_pt06b | 128 | 128 | 1e-6 | 16384 | 1 | 1.0 | 1.0 | forward_kl_topk_teacher_renorm | 128 | 2 | 2 | 20 |  | 42 |  |  | 2048 |
| skd_pt17b | 128 | 128 | 1e-6 | 16384 | 1 | 1.0 | 1.0 | forward_kl_topk_teacher_renorm | 128 | 2 | 2 | 10 |  | 42 |  |  | 2048 |
| trd_pt17b | 128 | 128 | 1e-6 | 16384 | 1 | 1.0 | 1.0 | forward_kl_topk_teacher_renorm | 128 | 2 | 2 | 20 |  | 42 |  |  | 2048 |
