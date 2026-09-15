# PERF-M4 — CPU/Metal 热态推理加速与内存压缩（最终报告）

日期：2026-09-14/15 · 机器：MacBook Air M4（4P+6E, 16GB, Metal 4）
团队：lead（决策/合并/验证）+ perf-cpu + perf-metal + perf-post（herdr-team 6z3naa）

## 0. 成果总表（同一窗口 interleaved A/B，load≈4）

| 维度 | main (11b00ae) | 合并树 (ws/test-merge@484fe5d) | 变化 |
|---|---|---|---|
| **CPU e2e t8/t4** (tiny, zh/03) | — | — | **无可测差异**（min/p10 符号相反 → 噪声，perf-cpu 判定）|
| **db_post（PPOCR_SUZUKI=1，6框图）** | 23.9 ms（同树 suzuki=0）| **2.7 ms** | **8.9×**，text 逐位相同，RSS 264→204MB |
| **db_post（19框图 zh/05）** | 45.7 ms | **3.0 ms** | **15.2×**，e2e cpu_min +3.7%（自洽真收益），RSS 244→198MB |
| **Metal e2e** (tiny) | ~101 ms（空机参考）| 同左（会话复用后 1-2ms/帧确定性节省）| SE 模型重写见 §3 |
| **CPU det_run**（裸 MNN cpu-time）| 684-767 ms | 502-519 ms | **-24~35%**（KleidiAI off）|
| **Metal RSS** | 198 MB | **145 MB** | **-27%** |
| **CPU RSS** | 253 MB | **216 MB** | **-15%** |

（注：本会话后期机器持续热节流，绝对值与早期测量不可比；只有同窗口同进程 interleaved 对比有效。perf-cpu 建立的 getrusage cpu-time + min/p10 方法论是所有 CPU 结论的依据；跨对/跨进程的单阶段 wall 时间一律不可引用。）

## 1. 已合并到 ws/test-merge 的分支

| 分支 | 内容 | 验证 |
|---|---|---|
| ws/perf-cpu (54439cb) | **KleidiAI 默认关闭**（PPOCR_MNN_KLEIDIAI=1 回开）；MnnSession 热路径（staging/readback tensor 复用、resize 跳过、去锁输出缓存）| 5 语言 MLC 与 main 逐位一致；prob-map bit-exact；cpu-time A/B 1.32-1.53x |
| agent/perf-metal (ca91c09) | Metal 会话精度注释+PPOCR_METAL_PREC=normal knob；commit 粒度 knob；版本号更正（3.6.1）；文档 | 跨进程 MD5 确定性；变形状逐字节一致；Metal zh 0.0153 |
| agent/perf-post (d238d8d..db9daf2) | **Suzuki-Abe 单遍轮廓追踪**（cv2 4.x/5.0 逐字节等价，39/39 掩码）；db_post 8.5-37x；diff_detpolys/verify_suzuki 工具 | **默认关**（PPOCR_SUZUKI=1 开启，见 §2）；开启后 en/03 v4_mobile cell 0.0898 FAIL（phantom blob）|
| ws/perf-lead (c9a78de, 5e9863a) | 相对 --model-dir 空 rec 字典 bug 修复；CLI PPORC_MNN_MODELS env 修复 | zh MLC 0.0151 不变 |

## 2. Suzuki 决策记录（为什么默认关）

新 tracer 是旧 tracer 的严格超集：恢复 46 个 paddle baseline 真框（recall 90.2%→94.9%），但也暴露 11 个 phantom blob（en/03）。Phantom 根因：**MNN prob map 与 paddle prob map 的 blob 级差异**（fp32 conv 数值路径不同），非 tracer 缺陷——恢复的 4 个真小框 poly 与 baseline **逐位相同**。无安全过滤器可分离 phantom（面积/短边与真框重叠）。811-cell gate 是硬契约 → 默认关闭，开关保留全部收益给能接受该 trade-off 的用户。真正修复方向：转换侧数值校准或 prob-map fidelity。

## 3. SE-pooling 模型重写（lead，未合并 — 需转换流程决策）

- 发现：det 图的 SE 全局池化 = `Pooling3D{isGlobal}`；MNN Metal `pooling_avg` 对 1×1 输出只派发 **1 个 threadgroup**（每 call 2.7ms 串行扫 W×H）→ det Metal GPU 时间 47-51%。
- 修复：`tools/rewrite_se_pool.py`：.mnn→JSON→`Reduction{MEAN,[2,3]}`→.mnn（weights 不动，纯算子重表达）。
- Metal 实测（空机窗口，3 次，方差<2ms）：**tiny 63→34.5ms（-45%）**；v6_small 88→56；v5_mobile 81→64；v4_mobile 83→60。**v4_server 1141→3541ms（回归，不重写）**——其 pooling 本就 grouped 派发（perf-metal 验证）。
- 数值：prob-map maxabs 1.25e-6~1.1e-2（输入相关）；**5 语言 MLC：CPU 与 main 完全一致；Metal zh/en/ja PASS（0.0153/0.0018/0.0280），ar/ru 与既有偏差逐位相同**；v5_mobile 重写 vs 原版 5 语言逐位相同（perf-metal 终验）。
- CPU 上重写为 2-6% 回归（perf-cpu：图节点 219→235，CPU 本就无 pooling 瓶颈）→ **必须按 backend 分派模型**（Metal 用重写版，CPU 用原版）。
- **已落地（task-7, commit 980e16d）**：`resolve_det_model_path()` 在 Engine 层按 backend 分派 —— Metal ∧ 命中允许清单（v6_tiny/v6_small/v5_mobile/v4_mobile）∧ `<name>.red.mnn` 存在才用变体；文件缺失静默回退原模型（部署无感）；GPU→CPU 回退会话固定加载原模型；损坏变体告警回退。`PPOCR_SE_REWRITE=0` 可关（A/B 用）。`include/ppocr/se_rewrite.h` 纯函数清单可单测（test_se_rewrite 4/4）。
- **验证更正（重要）**：早期"重写版逐位相同"的测量部分无效 —— 旧 CLI 的 --model-dir 默认值压掉了 PPORC_MNN_MODELS（5e9863a 修的 bug），两列实际都在评原模型。修正后的权威数字（perf-metal，真实分发）：v6_tiny+v5_mobile Metal 5 语言，4/5 语言逐位相同，最大 delta 4.1e-3（v5_mobile ja，≤1px 多边形位移），比 gate 低一个数量级；ru/ar 与原版完全相同（既有偏差）。我（lead）的早期 CPU 5 语言"完全一致"同理受污染作废；以本表为准。裸模型等价性（绝对路径直载）仍成立：maxabs 1.25e-6（真实图）/ prob-map 二值化一致率 100%。
- 变体文件当前位于 models/*.red.mnn（本机）；registry 分发（M3 流程）留作后续。注意 ensure_model 会删除与 registry size 不符的 <name>.mnn 覆盖文件 —— .red.mnn 命名不受影响（已验证）。

## 4. 既有问题清单（与本次改动无关，均已三项独立复现）

1. **ar/ru MLC 超标是既有偏差**：v6_tiny ru 0.0591 / ar 0.1443、v4_mobile ru 0.0888 / ar 0.1948、v5_mobile ru 0.0500 / ar 0.1726 —— 在未改动 main 上逐位复现。Linux CI 是否同样超标需核对（本机 baseline 为 canonical paddle 生成）。
2. `ar/02.jpg` 坏 JPEG（265835 extraneous bytes）→ rc=3（main 同样失败）。
3. MNN 3.6.1 Metal Winograd corrupt（ru/00: 6→37 boxes）与 fp16 storage 路径（MLC 0.1-0.6）都确认不可用。
4. MNN 上游缺陷：macOS 上 P/E core 分组是死代码（groups 只在 __linux__ 填充）；建议上游 issue（hw.perflevel0/1 sysctl）。
5. AGENTS.md 版本描述过时：submodule 实为 3.6.1（d407447），非 2.9.1。

## 5. 方法论沉淀（写入 PERF_M4_NOTES）

- 高负载/热节流下 wall-clock 完全失效（9x 抖动）→ **getrusage 进程 CPU 时间 + interleaved + min/p10**。
- Metal 计时：runSession 只提交；真实 GPU 时间在 readback（waitUntilCompleted）。400px 小输入噪声 17%（同计算重打包都能测出），只有 full-res（≥1053px）信噪比可用。
- 单语言 gate 不足以判数值路径改动（zh bit-exact 而 ru 崩）→ 最少 zh/en/ja/ar/ru。

## 5.5 更正与澄清：Metal "GPU 利用率" 的正确量纲

- 最终总结中"1%→7% FLOP 峰值利用率"的 7% 是算术错误；正确值为 ~1-1.5%。
- 但 FLOP 峰值对这种负载失灵：34.5ms 中 conv 类仅 ~17.5ms，其余为 Raster/Binary/Unary 等纯访存算子（激活流量估算 2-4GB → 带宽地板 17-33ms @120GB/s）。按带宽口径利用率约 30-60%，距硬件极限 1.5-2×。
- 该图在 Metal 的现实地板 ≈ 20-25ms。击穿需：fp16（精度 gate 否决）、算子融合（MNN 上游）、降输入分辨率（基线契约变更）。

## 5.6 CoreML 实验记录（事后补录，全部实测 M4 / MNN 3.6.1 / macOS 26 SDK）

- **可行性**：重编 libMNN（`-DMNN_COREML=ON`，build_coreml/）后 `--backend coreml` 可用。det 固定尺寸路径完全可用，热态 **10.6ms**（vs Metal ~35ms GPU、CPU 99ms）。
- **rec 不可用**（结构性）：rec 模型动态宽度 [1,3,48,-1] vs CoreML `EXACT_ARRAY_MAPPING` 固定形状 → "Failed to Invok the Model"。已实现混合调度：det=CoreML、rec=Metal（load 失败回落 CPU），`PPOCR_COREML_MIXED=0` 可诊断。
- **硬伤 1：每新形状 ~0.5-0.8s 编译费**。CoreML 后端每次 resize 都磁盘写 mlmodel + compileModelAtURL。变尺寸 eval 集 0.84s/张，比 Metal 慢一个量级。仅固定尺寸流（视频）能吃到 10.6ms。
- **硬伤 2：数值退化**。ru MLC 0.0591→0.1596（新回归，ru/04 整图 miss），ja 0.028→0.046（翻倍但<0.05），zh/en 干净。根因指向 ANE fp16（prob-map checksum 10.0 vs CPU 52.6，二值化只吸收了部分）。
- **硬伤 3**：`MLComputeUnitsAll` 硬编码于 MNN，无法强制避开 ANE（submodule 禁改）。
- **定位**：opt-in 快速路径，默认 AUTO→Metal 不变。e2e 最快 27ms（37 FPS，det CoreML + rec Metal + Suzuki）。
- **附带修复**：CommandLineTools 27.0 SDK 的 tbd 文件损坏（unknown architecture），build/ 需 `-DCMAKE_OSX_SYSROOT=.../MacOSX26.sdk` 重配才能链接。

## 6. 未竟事项 / 后续任务

1. SE-rewrite 合并落地（backend 分派 + 811-cell Linux 复验）— 最大单项 Metal 收益。
2. Suzuki 的 prob-map fidelity 修复（真正解锁 -87% db_post + 4.7% recall）。
3. det 输入分辨率（limit_min 契约变更，需全矩阵重基线）— CPU/内存的终极杠杆。
4. Metal e2e 空机终测 + bench.py 协议更新（cpu-time 模式）。
5. cls test_cls 在 M4 上的 env 失败（perf-post 报告，pre-existing）。
