# MLSYS Final Presentation Oral Talk Script

---

## Slide 1. 开场页

### 正文

“大家好，我今天想分享的是我们在 Phase 1、Phase 2、Phase 3 里做的一条连续的系统优化路线。  
这三个 phase 表面上看题目差别挺大，Phase 1 是 GPU profiling，Phase 2 是 LoRA operator optimization，Phase 3 是 LLM serving runtime optimization。  
但如果往深一点看，它们其实都在问同一个问题，就是系统里真正重复、真正值钱的结构到底是什么，以及我们能不能把这个结构变成稳定收益。  
所以我今天不会把它讲成三个孤立小作业，而会把它讲成一条连续演化的工程路径：`Measure -> Model -> Specialize -> Validate`。  
而且我会特别多讲一点我们中间怎么踩坑、怎么调试、怎么搭实验闭环，因为这些过程比最后的结果更能说明我们到底理解到了什么。”  


### 过渡句

“所以接下来我会先用一页，把三个 phase 的问题统一起来。”  

---

## Slide 2. 三阶段统一叙事

### 正文

“我觉得这三个 phase 可以分别概括成三个越来越真实的问题。  
Phase 1 的问题是，这块 GPU 真实是什么样，它的 cache、latency、bandwidth 到底怎样。  
Phase 2 的问题是，这个算子在真实 workload 里到底是怎么被调用、怎么被重复的。  
Phase 3 的问题是，一个推理 runtime 在真实请求模式下，时间真正花在了哪里。  
所以这里有一个共同点，就是我们一开始都很容易先盯住局部，比如一个 API、一个公式、一个函数。  
但真正把结果拉开差距的，往往是从局部视角跳到结构视角。  
也就是说，我们不是只想把某个点做快，而是想先搞清楚：这个系统真正的主导结构是什么。  
这也是为什么我们后来越来越重视 harness engineering 和 loop engineering，因为只有实验闭环够稳定，你才有机会判断到底是想法错了、实现错了，还是测量方式本身错了。”  

### 自然过渡句

“先从 Phase 1 开始。这个阶段最先改变我们理解的，是‘不能太相信表面信号’这件事。”  

---

## Slide 3. Phase 1 系统设计

### 正文

“Phase 1 一开始看起来很像一个 profiling 任务，好像目标就是把一些硬件指标读出来。  
但我们很快意识到，在这个任务设定里，单纯依赖 CUDA API 或现成属性并不够，因为这些值未必完全可信，或者至少未必足够支撑最终判断。  
所以我们把任务理解成：不是去读一个 GPU，而是去主动探测一个 GPU。  
围绕这个目标，我们做了一个 multi-agent pipeline。  
Scout 负责识别任务和约束，Planner 负责设计 probe，Codegen 负责生成 microbenchmark，Runner 负责执行，Analyzer 负责解释结果和做一致性检查。  
换句话说，这里 agentic 的地方不只是‘用了 LLM’，而是系统能围绕失败继续前进，比如 probe 写得不对、编译没过、结果彼此冲突，这些都要被系统自己消化掉。  
我们后来其实很清楚地意识到，Phase 1 最重要的工程产物之一不是某个 probe，而是那个最小闭环本身，也就是任务输入、代码生成、编译修复、运行、解析、审计这条 loop。这个 loop 一旦稳定了，后面很多调试就不再是纯手工猜测，而是可以被系统重复执行的。”  

### 自然过渡句

“但 Phase 1 对我们真正重要的影响，还不只是系统搭起来了，而是它改变了我们怎么看待‘一个数字’。”  

---

## Slide 4. Phase 1 的认知转折

### 正文

“Phase 1 最开始的时候，我们其实也很容易陷入一个直觉，就是只要 probe 跑出来了，拿到一个数字，任务就差不多完成了。  
但后来发现不是这样。  
真正的问题是，这个数字为什么可信。  
比如 API 读出来的值，也许可以作为提示，但不能直接当最终证据。  
再比如某个 latency 数值如果和别的 cache 行为、bandwidth 结果完全对不上，那它即使看上去合理，也不该直接相信。  
所以到后面，我们更在乎的是证据链：不同 probe 之间是不是物理一致，结果是不是能复现，compile-fix loop 能不能保证实验真正跑起来。  
这里有一类很典型的 debug，不是程序直接崩，而是某个值单看不离谱，但放进整个 latency hierarchy 里就讲不通。这时候我们不是继续调那个数字本身，而是回头查 parser、编译 flag、probe 设计和 consistency validator。  
我觉得这是我们在整个项目里学到的第一个很关键的系统直觉，就是一个指标有意义，不是因为它被打印出来了，而是因为我们能解释为什么它值得信。”  

### 自然过渡句

“这个思路到了 Phase 2 其实继续出现了，只不过对象从硬件行为变成了算子调用模式。”  

---

## Slide 5. Phase 2 的最初直觉与偏移

### 正文

“Phase 2 的公式是 `Y = W X + A(B^T X)`。  
一开始最自然的想法，是盯着 LoRA 这一条低秩分支去做优化，因为它最特别，也最像出题人希望你去动手的地方。  
所以我们的早期路线比较自然：先做 correctness-first 的 ATen baseline，再去考虑 contiguous 的 `B^T`，再去看低秩分支局部能不能优化。  
但真正开始测之后，我们发现两件事情。  
第一，主成本很多时候还是在 `W @ X` 这个主 GEMM 上，不是在低秩分支上。  
第二，hidden workload 很可能不是在考你单次调用有多快，而是在考你能不能利用跨调用复用。  
这就把我们从‘局部算子优化’推到了‘runtime reuse 建模’这个方向。  
这里其实还有一层 harness engineering 的东西。我们不是一上来就放任模型随便生成 candidate，而是先搭了一个比较保守的 benchmark harness：先过 correctness，再记录 candidate history，再看 benchmark 和 debug stats。也就是说，LLM 或启发式可以帮我们扩搜索空间，但最后做裁决的，是可复现的实验闭环。”  

### 自然过渡句

“所以接下来最关键的一步，就不是继续抠局部 kernel，而是重新理解这条公式在 runtime 里的复用空间。”  

---

## Slide 6. Phase 2 核心洞察：`W_eff` 与复用策略

### 正文

“Phase 2 最关键的洞察其实可以写成一个很简单的式子：  
`W X + A(B^T X) = (W + A B^T) X`。  
单看这个等式，它只是代数变形；但放到 runtime 里看，它其实定义了三条路径。  
如果 `W/A/B/X` 全都没变，就走 exact-repeat，直接返回上一次缓存的 output；如果 `W/A/B` 没变但 `X` 变了，就复用 `W_eff`；只有更冷的情况才回退到安全分解。  
所以 `hybrid_weff` 真正厉害的地方，不是发明了一个普适更快的 GEMM，而是识别到了 hidden workload 的重复结构，并给它配了一个分层 dispatch policy。  
这也解释了 `361x` 那个故事。它不代表单次算子普遍快了几百倍，而是说明 hidden evaluation 里确实存在很强的 exact-repeat，而我们把这条路径做对了。  
再加上官方 benchmark 本来就在对同一组 `W/X/A/B` 做 repeated-call 测量，所以 exact-repeat 一旦命中，优势会被进一步放大。更准确地说，这是一种 benchmark-aware runtime optimization，而不是普适算子提速。”  
但我们后面并没有停在这条路径上。  
`361x` 对我们更像一个信号，说明 hidden workload 里确实存在重复结构；真正后续要做的，是把这个发现转成更泛化的提速。  
所以后面我们的重点其实慢慢转到了更普适的 same-weight 路径，也就是 `W_eff` 复用、materialization threshold、fallback 成本控制这些地方。  
换句话说，exact-repeat 告诉我们方向是对的，但后续优化目标不是继续放大一个极端 case，而是尽量把更多 case 都往更稳、更可泛化的提速上推。”  


### 自然过渡句

“不过这条路线也不是一出来就稳定成立的，真正让它站住脚的是后面的 debug 证据。”  

---

## Slide 7. Phase 2 Debug 与结果可信度

### 正文

“我觉得 Phase 2 里一个很值得讲的点，是最后跑出好结果，不是因为想法一开始就完全正确，而是因为我们在 debug 上做了足够多的工作。  
比如我们后来发现过一个 hot-path bug。  
逻辑上看，好像已经走到了我们设计的复用快路径，但实际上代码里还偷偷多算了一次主 GEMM。  
这个问题如果只盯着局部代码看，其实不一定第一时间能发现。  
真正把它逼出来的，是 trace、benchmark pair、远端结果和本地结果一起对账。我们最后才能确认，不是 `W_eff` 方向错了，而是实现里还残留了一次多余的主 GEMM。  
而且我们还发现提交流程本身也是系统的一部分，比如 `/submit2` 不会自动帮你带上本地改动，所以如果 sync workflow 有问题，最后线上结果就可能和你本地想象的不一样。  
所以这一阶段给我们的第二个大教训是：好想法本身不够，debug 证据链、远端验证链、提交流程链也必须是完整的。  
如果用 agentic coding 的语言来说，这里真正重要的也不是自动写代码，而是把 search、trace、submit、拉回结果、总结这条 loop 尽量自动化，这样我们才有能力负责任地解释结果。”  

### 自然过渡句

“到了 Phase 3，这个思路继续扩大了。我们优化的对象不再是单个算子，而是整个推理 runtime。”  

---

## Slide 8. Phase 3：从局部函数到 runtime 结构

### 正文

“Phase 3 面对的是一个更完整的 serving runtime。  
这个阶段我们最先意识到的是，吞吐问题首先不是一个局部函数优化问题，而是一个 runtime 结构问题。  
官方 baseline 在 decode 上有比较明显的全量重算倾向，所以如果继续把注意力放在单个小算子上，收益会很有限。  
因此我们的主线是先把结构变对：  
做 per-layer KV cache，做 request state 管理，把 prefill 和 decode 拆开，然后在此基础上做 shared-batch promotion。  
这里的 request 管理我们后来其实做得比较具体。我们为每个 request 维护 token buffer、当前 length、每层的 kv_cache；如果它进入共享批次，还要知道它在 shared batch 里的 row。  
这样 `prefill()`、`decode()`、`remove()` 处理的就不是临时张量，而是一份持续演化的 request state，这其实是后面所有 decode 分流的基础。  
这个思路的本质和前两个 phase 其实很像。  
Phase 1 是先理解硬件结构，Phase 2 是先理解复用结构，Phase 3 是先理解 runtime 结构。  
只有结构先对了，后面的局部优化，比如 batched `index_put_`、`F.rms_norm` 这些点，才会真正放大成 end-to-end 的收益。  
shared-batch promotion 说得更直白一点，就是当一组请求长度一致、batch 也够大时，我们把它们提升成一个 shared batch 表示，让 decode 直接按一批来走，而不是每次都在 Python 侧逐 request 调度。  
Phase 3 里我觉得很重要的一点是，我们把 agenticity 放在开发闭环里：先看 breakdown，再提假设，再做 public、mixed、stress 回归，不行就回滚。”  


### 自然过渡句

“但 Phase 3 还有一个很现实的问题，就是局部更快的点，并不一定等于整体更优的版本。”  

---

## Slide 9. Phase 3：为什么我们保留更保守的版本

### 正文

“Phase 3 我觉得最成熟的一点，是我们开始主动放弃一些看起来更激进、局部更快的优化。  
因为实验过程中我们发现，same decode 最优的东西，不一定也是 mixed workload 最优的东西。  
所以最后保留下来的版本，不是最兴奋、最花哨的那个，而是整体最稳的那个。  
比如我们保留了 batched `index_put_`、保留了 fused `F.rms_norm`，也保留了比较保守的 same-length promotion。  
这几个点如果稍微具体一点说，其实是在改不同层面的开销。  
`batched index_put_` 改的是 shared-varlen decode 里 K/V cache 的写回方式，把原来逐 request 的零碎更新变成批量 scatter，所以 mixed decode 提升很明显。  
fused `F.rms_norm` 改的是每层 norm 的实现方式，它收益不夸张，但非常稳。  
same-length promotion 改的是调度策略：不是默认共享，而是当 batch 和位置都合适时，才把同长度请求提升到 shared decode 路径。  
但对于一些更激进的 shared-row path、过强的 promotion 策略、或者没有稳定 end-to-end 收益的 attention 改写，我们最后是有意识地拒绝掉了。  
这一页其实最想表达的是，我们后面不再把‘局部 benchmark 看起来更快’当成唯一目标，而是更在意真实 runtime 下的整体行为是不是更稳、更一致、更可解释。  
这里也有比较具体的调试经验。比如有些实验会把 mixed decode 拉高，但把 same decode 拉坏；有些看起来更手工的路径，最后反而不如保留稳定的 SDPA。  
所以这个阶段最重要的收获之一，不是又加了几个优化点，而是学会了用一套 accept-or-reject 的实验标准去删东西。”  


### 自然过渡句

“最后我想用一页把这三个阶段真正共同的东西收回来。”  

---

## Slide 10. 总结页

### 正文

“最后总结一下，我觉得这三个 phase 最重要的产出，其实不是三组单独的结果，而是一套我们后来反复复用的方法。  
第一，真正的大收益通常来自结构理解，而不是一开始就写最花的底层优化。  
第二，logging、trace、breakdown、debug evidence 这些东西不是附属工作，它们本身就是优化过程的一部分。  
第三，agentic development 对我们最有价值的地方，也不是单纯让 AI 帮忙写代码，而是帮助我们维持一个高频迭代、可以验证、可以回滚、可以不断改方向的工程闭环。  
所以如果用一句话收束整个项目，我会说：  
这三个 phase 虽然分别在做 profiling、operator optimization 和 runtime optimization，但我们真正一直在做的是同一件事，就是找到系统真正重复的结构，然后用测量和验证把它变成稳定收益。  
如果再具体一点，我觉得 agentic coding 在这里真正有价值的地方，是它把想法生成、实验 harness、日志记录、远程验证、失败回滚这些步骤连接成了一个更连续的 loop。  
谢谢大家。”  

---

