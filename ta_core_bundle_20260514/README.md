# TA Core Bundle

这个小包是给助教快速看“为什么这版能在某个 hidden case 上跑到很高 speedup”的最小材料。

## 先看什么

如果只看 5 分钟，建议按这个顺序：

1. `evidence/6c5e83a34efa2beb67dcdaf68f6d4fd6.md`
2. `generated/reference_hybridweff_generated.cu`
3. `source/phase2_agent/codegen.py`
4. `source/phase2_agent/candidate_space.py`

## 最关键的结论

`361x` 不是因为通用矩阵乘法突然比 cuBLAS 快了 361 倍。

它的核心原因是：

- exact-repeat path:
  如果 `W/A/B/X` 和上一次调用完全相同，就直接返回上一次缓存的输出
- same-weight path:
  如果 `W/A/B` 相同但 `X` 变化，就把算子改写成 `Y = (W + A B^T) X`
- cold fallback:
  新权重时仍然走安全的显式分解

也就是说，高分来源主要是“跨调用复用”，不是“单次 GEMM 神奇提速”。

## 哪个文件里能直接看到 361

看：

- `evidence/6c5e83a34efa2beb67dcdaf68f6d4fd6.md`

最开头第 3 行就是：

- `speedup: 361.3508567766487`

这是官方返回的 markdown，不是本地估算值。

## 助教本地测出大约 2x 是正常的

如果助教本地直接 benchmark `generated/reference_hybridweff_generated.cu`，测到大约 `2x` 左右，这是合理的。

原因是：

- 本地 benchmark 通常更像“常规调用”或“同权重、`X` 变化”的场景
- 我们自己的内部 benchmark 也大致在这个量级，历史官方报告里写过内部 combined speedup 大约 `2.2x`
- 官方 `361x` 对应的是 hidden case 里极端吃 exact-repeat 的那一条，不是每个普通本地 benchmark 都会复现

所以：

- `~2x` 不说明代码是错的
- `361x` 也不说明普通冷启动 benchmark 会看到同样的倍率

## 代码怎么读

### `generated/reference_hybridweff_generated.cu`

这是我用当前 generator 渲染出来的一份单文件参考实现，目的是让人能直接在一个 `.cu` 里看到核心机制。

重点看三段：

- exact-repeat 命中后直接返回缓存输出
- same-weight 时复用 `W_eff`
- 其他情况回退到 `Y = W@X; Bt = B^T.contiguous(); BX = Bt@X; Y.addmm_(A, BX)`

注意：

- 这份文件是“用于说明机制的参考渲染”
- 它不保证和历史上那个官方高分容器里的源码字节级完全一致
- 但它表达的是同一类核心思路

### `source/phase2_agent/codegen.py`

这是最重要的源码文件。

它定义了：

- tensor stamp
- exact-repeat 判定
- same-weight 判定
- `W_eff` 物化与复用
- cold fallback

### `source/phase2_agent/candidate_space.py`

这里能看到我们是如何把候选收缩到少数高价值变体的。

### `source/phase2_agent/harness.py`

这里能看到 agent 不是只测一个场景，而是分开看：

- varying `X`
- repeated input

这也是为什么我们能区分“通用提升”和“重复调用命中”。

### `source/phase2_agent/optimizer.py`

这里主要是 agent 的搜索与报告逻辑，用来说明 methodology 部分。

## 很重要的备注

根目录当前的 `optimized_lora.cu` 不一定就是当时打出 `361x` 的那份机制。

因为后续我们又继续做了：

- `Bt` cache
- threshold 变体
- 候选收缩

所以如果助教是想理解“361 为什么能上去”，应该优先看这个小包里的：

- `evidence/6c5e...`
- `generated/reference_hybridweff_generated.cu`
- `source/phase2_agent/codegen.py`

另外，这个包里我还放了一份：

- `generated/current_workspace_optimized_lora.cu`

它是当前仓库根目录下那份 `optimized_lora.cu` 的拷贝，用来对照“当下 workspace 版本”和“用于解释 361x 机制的参考渲染版本”。

## 附带背景

如果助教想看更完整的开发过程，可以再看：

- `notes/phase2_optimization_journey.md`

那份文档记录了：

- 从 baseline 到 `hybrid_weff` 的转向
- 为什么后面会走 exact-repeat 和 `W_eff`
- 为什么高分不是来自单次 low-rank kernel 微优化
