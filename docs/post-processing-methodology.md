# 现代顶尖摄影师后期方法论：工程化参考

本文把公开可核验的后期方法论，蒸馏成 SuperPhoto 能落地的原语、顺序、约束和验收规则。
它不是风格滤镜清单。目标是：同一张 RAW 在不同体裁、不同伦理制度下，走同一套管线，只切换政策与配方。

对照代码：`pipeline.py` 的 restore → critique → retouch；`workflow.py` 的两轴；`recipes.py` 的 MOVES；`qa.py` 的安全分；`layers.py` 的分区蒙版。

收集范围：风光 / 人像美妆 / 婚礼胶片 / 电影调色 / 商业时尚 / 纪实伦理 / 野生动物 / 电商产品 / 开源 scene-referred。
时间范围以 2015–2026 年公开课、书籍、厂商手册、竞赛规则为主。安塞尔·亚当斯分区曝光作为数字流程的历史骨架保留。

---

## 0. 怎么读这份参考

摄影师口头上的「先拉高光」「先磨皮」「先套胶片」经常互相打架。工程上只保留能写成规则的部分：

| 能工程化 | 不能工程化（只作风格口味） |
|---|---|
| 操作顺序、冻结轴、回滚门 | 「有感觉再停」 |
| 按亮度/语义分区，而不是全局一刀 | 某张代表作的具体滑杆数值 |
| 色相锁定（草还是绿、天还是蓝、皮还是橙） | 青橙电影感的饱和幅度 |
| 安全验收：裁切、色漂、塑料皮、光晕 | 「更有艺术性」 |
| 伦理制度：允许改什么、禁止改什么 | 个人审美偏好 |

现有工程已经做对的三件事，后面所有流派都在重复：

1. **修复与美化分轴**（`workflow.py`：restoration 不改颜色与整体明暗；retouch 不做去噪/锐化）。
2. **校正与 look 分离**（`preference.py`：`correction.merged(look_recipe)`；README 也写了这一点）。
3. **安全分压过存在感**（`qa.py`：safety 权重 0.58；太强就回退）。

下面先给一张「几乎所有顶尖流程都同意」的共识表，再按流派展开，最后映射到 SuperPhoto 缺口。

---

## 1. 跨领域共识（可直接当管线不变量）

### 1.1 先诊断，再动手

Cullen Kelly 把 Walter Murch 的「牵涉痛」（referred pain）搬到调色：你想提亮主体，真正该做的常常是压背景。
Pratik Naik 的原话更硬：**先判断这张图什么不该修**，比学会频率分离更重要。
Ted Gore 在构图与色相上同一原则：自然界有「情境预期」（context of nature）——天是蓝的、草是绿的；把草拧成青色，观众会先问「草为什么是青的」，而不是欣赏和谐。

工程含义：`critique` 必须输出「问题真正在哪」和 `do_not`，而不是一份滑杆愿望清单。现有 `Critique.do_not` 已经是这个槽位。

### 1.2 校正（primary）在 look 之前，且校正必须可单独验收

几乎所有严肃流程都把工作分成两段：

| 段 | 问的问题 | 典型操作 | 谁在教 |
|---|---|---|---|
| Primary / 校正 | 这张图准不准、干净不干净 | 白平衡、曝光、高光保护、黑白场、镜头与噪点 | Capture One 官方、Cullen Kelly、Scott Kelby、Andy Hutchinson、darktable |
| Look / 美化 | 这张图像不像我们要的味道 | 胶片曲线、青橙、低键、HSL 风格、颗粒暗角 | Mastin、Strohl、电影调色、现有 XMP looks |

例外只有一个，而且是有物理理由的：Mastin Labs 的胶片预设会改变高光滚降，所以他们要求**先套胶片特性，再拉曝光**。那不是「先风格后校正」，而是「先换显示变换，再在新的高光行为里定曝光」。见 §3.5。

SuperPhoto 当前把 correction 和 look **加性合并后再** `split_retouch_recipe`。这对 Lightroom 滑杆风格的 look 够用；对「改变高光肩部」的胶片/LUT 不够。缺口见 §6。

### 1.3 场景线性里做光，显示空间里做味

darktable 的 scene-referred（filmic / sigmoid）和 Cullen Kelly 的 DaVinci Wide Gamut / ACES 说的是同一件事：

- 曝光、白平衡、局部明暗，发生在 scene-referred（线性、场景亮度）。
- 对比度肩部、胶片打印感、LUT，发生在一次**唯一**的显示变换里。
- 显示变换模块不要扛全部工作（darktable 手册原话：*less is more*）。

SuperPhoto 已在线性空间 ingest（`load_linear`），渲染时再 `linear_to_srgb`。缺的是显式的「显示变换节点」：现在 look 是一组 display-referred 滑杆的加和，不是一条 tone-mapping 曲线。

### 1.4 全局定骨架，局部塑光；对比度尽量不要全局猛拉

Bruce Percy：新手把对比度加在全局，眼睛会被到处拉，图会「看久了累」。
Nick Page：闪避/加深（dodge & burn）才是把二维照片拉回三维的主工具。
Ansel Adams 暗房的闪避加深，到数字时代没有消失，只是改名叫亮度蒙版、范围遮罩、画笔。
Thomas Heaton：不要默认高光 −100、阴影 +100；对比用压高光 + 压黑场，清晰度打在主体径向，而不是全局饱和。

工程含义：`Recipe.contrast` 只能做轻量骨架；真正的「层次」应走 `LocalEdit` + 亮度条件。

### 1.5 饱和度跟着光走，不跟着滑杆走

Ted Gore、陈建强、中文风光教材里的「通透」口径高度一致：

- 直射光区域：可以较高亮度、较高饱和。
- 反射光区域：中等。
- 无光区域：低饱和、偏暗。
- 全局拉饱和 = 「核辐射日落 / 核绿叶子」（landscape 教程里的 *nuclear sunset / nuclear greens*）。

Heaton 明确拒绝全局 saturation blast，只用少量 vibrance。
Gore 的绿：降饱和、略偏橄榄、略压暗——不是把绿拧成别的色。

工程含义：HSL 与 vibrance 应乘以亮度权重；绿/蓝有色相走廊锁。现有 `gore_greens` 是这条规则的手工版。

### 1.6 修复类操作必须色中性，且发生在创意之前

2024–2026 的主流顺序几乎锁死：

1. DxO PureRAW / Lightroom Denoise / darktable denoise：**在原 RAW 上、裁切之前**。
2. Glyn Dewis 把 Lightroom AI 工作流写成 **Prepare → Repair → Finesse**（先 demosaic/denoise/super-res，再去污，最后 look 与蒙版）。否则 AI 蒙版会和像素状态脱节。
3. 野生动物：Anthony Morganti / Matt Kloskowski / DxO 文档都强调——未裁切时去噪，锐化只打在羽毛/皮毛/眼睛，虚化背景不要锐。

这与 SuperPhoto 的 Axis A（restoration）完全同构。现有门控（chroma drift ≤ 0.008、tone drift ≤ 0.03）应视为不变量，不要为了「更锐」而放松。

### 1.7 步骤要少，意图要少

| 体系 | 上限 | 内容 |
|---|---|---|
| Scott Kelby 7-Point | 7 步 | 配置文件 → 白平衡 → 曝光/白黑场 → 对比 → 局部 → 颜色 → 收尾 |
| Mastin 3-Step | 3 步 | 套胶片 → 曝光 → 白平衡（偶尔扫描 soft/hard） |
| SuperPhoto retouch | 最多 3 个高阶意图 | tone / color / atmosphere / subject / background |
| Andy Hutchinson Universal | 5 段 | Prep / Light / Structure / Color / Finish |

「步骤少」不是偷懒：每多一个全局滑杆，就越难回滚、越难解释、越容易和 look 打架。
现有 `split_retouch_recipe(max_attributes=3)` 与这条共识对齐。风光若需要第 4 个意图，应是「局部塑光」，不要把全局 contrast+clarity+dehaze+sat 一次性打开。

### 1.8 输出锐化 ≠ 创意锐化；磨皮 ≠ 去纹理

美妆：Julia Kuzmenko / Pratik Naik —— dodge & burn 匀色阶、保留毛孔；频率分离是最后手段，不是默认。
风光：输入锐化（raw sharpen）在前期，输出锐化按媒介（屏 / 印）在最后。
纪实：World Press Photo 2026 直接禁止 Super Resolution / Topaz 一类「生成新信息」的放大锐化。

SuperPhoto 已把 `subject_sharpen` / `clarity` 从 retouch 轴剥掉。这是对的。缺口是：editorial 类 look（`editorial_snap` 含 clarity）在 retouch 里会被清零，杂志硬朗目前几乎只靠 contrast。见 §6。

---

## 2. 一条可落地的标准管线（八段）

把上面的共识压成 SuperPhoto 可执行的阶段。现有代码已经覆盖其中两段；其余是显式化，不是推倒重来。

```
0 制度与体裁     ethics regime + genre   （现在只有 genre）
1 修复           restoration             （已有，色中性）
2 几何           crop / horizon          （已有 GeometryHint）
3 一次校正       primary grade           （现嵌在 correction 里）
4 局部塑光       dodge/burn + luma       （LocalEdit 的雏形）
5 物体色         HSL / skin lock         （Recipe.hsl）
6 显示变换/Look  filmic / LUT / XMP      （现为滑杆加和）
7 气氛收尾       grain / vignette        （atmosphere 组）
8 验收回滚       QA gates                （已有，可加锁）
```

顺序约束：

- 1 必须在 3–7 之前，失败则整段丢弃（现有 `score_restoration`）。
- 3 必须在 6 之前，除非 look 被标记为 `changes_highlight_rolloff=true`（胶片特性），此时 6 的 tone-map 先上，再在新空间里微调曝光。
- 4 的对比度变化优先于 3 的全局 contrast。
- 7 不得回写到 3 的白平衡。
- 8 按制度切换门槛，不是按 look 切换。

---

## 3. 流派目录（按可工程化程度）

每位只保留「步骤、冻结、禁令、可映射字段」。具体滑杆数是口味，不进默认配方。

### 3.1 风光·克制写实

**Thomas Heaton**（YouTube 后期系列、Adobe Live）

- 镜头校正 + 去色差 → 裁切（竖图常用 4:5）→ 适量压高光、少量提曝光、压黑场做对比 → 径向/渐变 + **亮度范围遮罩** 打在主体 → 少量 vibrance，不用全局 saturation。
- 原话级原则：不要默认高光 −100 / 阴影 +100；阴影可以比较暗，主体才是山。
- 清晰度打在主体，不打在天空。
- 映射：现有 `heaton_tone`（highlights −0.18, contrast +0.16, whites +0.05, blacks −0.06）是骨架；缺的是「径向 + 亮度范围」这一局部原语。

**Alister Benn / Expressive Photography**

- 五个表达元素：光亮度、对比、色彩、几何、气氛。后期是把情绪关系写清楚，不是把动态范围填满。
- 2014 年后多数单张曝光，少合成。口诀：impact versus ego（冲击对自我）。
- 黑白：先在彩色空间把通道关系做好，再转单色。
- 映射：气氛（atmosphere）与几何是一等公民；不要用 HDR 合成冒充「表达」。

**Bruce Percy**

- 简化：形与影调梯度，而不是「更多东西」。
- 对比是局部的：前景/背景影调分离；低对比 = 平静，高对比 = 紧张。
- 颜色数量过多会视觉疲劳；近年从「几乎去色」摆回「有节制的颜色」。
- 映射：QA 可惩罚「太多高饱和色相同时存在」；局部 contrast 优于全局 contrast。

**Nigel Danson / James Popsys / 克制派 YouTube 风光**

- 单张 RAW、少合成、少 Orton、少「糖水」。
- 与 Heaton 同族：保护现场光的方向，而不是重画一张日落。

### 3.2 风光·亮度蒙版与表达派

**Ted Gore**

- 色相只在「还是那种东西」的走廊里微调（草绿可以偏黄绿或蓝绿，不能变青）。
- 双处理同一份 RAW：一份管影调/对比，一份管颜色，再蒙版合成。
- 闪避加深塑造体积；可以加强雾气与光，但不把湖画到山前。
- D810 时代他甚至偏向略欠曝（expose to the left），因为阴影可救、假色不可忍。
- 映射：现有 `gore_greens` + `warm_light_cool_shadow` + `sky_settle`。应升级为「色相走廊锁」+「饱和度 × 亮度」。

**Sean Bagshaw + Tony Kuyper（TK 面板）**

- Lightroom 只做：透视、色差、输入锐化、白平衡、去尘，并同步到包围曝光。
- Photoshop：亮曝光在底、暗曝光在上，用亮度选区把正确曝光的像素画进去。
- 亮度蒙版不是滤镜，是「按明度取数」的选区生成器。Lights 1–6 / Darks 1–6 是离散的 luma 区间。
- 映射：SuperPhoto 的 `layers.py` 有语义分区（主体/天空/植被），**没有 luma 分桶**。这是风光深度的最大缺口。

**Nick Page**

- Dodge & burn 是后期最强工具：深度、气氛、视觉路径、色彩分离。
- 自由手绘 + 引导式（沿结构）+ 智能对象 + 对比调整四种实现。
- 映射：`LocalEdit` 应增加「沿主体结构的加光/压暗」，而不是只有 `background_darken`。

**Marc Adamus**

- 用过渡引导视线：暗→亮、冷→暖、大→小、饱和→欠饱和、虚→实。
- 曝光合成 + 焦点合成是为了克服媒介极限，不是为了超现实。
- 自创 clone-paint / history-paint：把「我想要的光」画上去，一次一张图成百上千笔。
- 工程上不要模仿他的笔数；要模仿他的**过渡向量**：每张图有一个视线目标，所有局部对比/冷暖都指向它。

**Daniel Kordan**

- 旅行向、快：库管理 → 全局 → 局部渐变 → 标志性柔化/高光辉光 → 导出色空间。
- 辉光/Orton 是气氛，不是默认。应受 `atmosphere` 配额限制。

**Elia Locardi**

- 反对「HDR 味」：包围曝光是为了动态范围，不是为了把所有滑杆推到 10。
- 手工曝光混合 + 亮度蒙版压天空，阴影不一起被压。
- 现场用渐变镜减少后期混合量。
- 映射：高动态范围场景优先「分区曝光」而不是全局 shadows +100。

**Greg Benz / Lumenzia、Jimmy McIntyre / Raya Pro、Blake Rudis**

- 把 TK 传统产品化：亮度/饱和度/色相三套选区，Orton，光晕控制。
- 工程价值：选区生成是基础设施，不是某个摄影师的秘方。

### 3.3 风光·中文体系（与 Gore/Heaton 同构，术语不同）

**陈建强《摄影金奖之路·风光》**

- 口诀：该亮的亮、该暗的暗、该冷的冷、该暖的暖。
- 直射光 / 反射光 / 无光 三区决定饱和与亮度上限。
- 后期准则：艳而不俗、控制溢出、冷暖对比用分离色调而不是全局色温死拧。
- 「通透」：中灰层次打开、黑场不并死、雾霾有控制、饱和不爆。

**「Thomas 看看世界」《风光摄影后期基础》**

- 前期是谱曲，后期是演奏。
- **局部处理是后期的核心思想。**
- 现代风光六大合成：接片、HDR、焦点、时间、透视、（及关联的）焦段混合。
- 避免糖水片 = 避免全局对比+饱和+清晰度三连。
- Camera Raw 顺序：光与色的基本面板 → HSL → 分离色调 → 局部滤镜 → 效果 → 降噪锐化。
- 三步锐化：质感锐化、（创意）、输出锐化。

这些和 SuperPhoto 的「最多三个美化方向」高度同构。中文用户对「通透」的口头验收，应写成 QA：中灰标准差上升、高光不裁、饱和增益有顶、天空不品红。

### 3.4 探险 / 商业旅行 look

**Alex Strohl（AS1 / AS2 预设）**

- 工作方式：先定曝光和白平衡（预设故意不碰这两项），再套 look。
- look 特征：高光到阴影的短过渡（戏剧对比）、颜色在现场色上推，不造假色、颗粒与纹理、径向突出小人。
- 商业片还要尊重服装品牌色（他原话：夹克颜色不能被 look 改掉）。
- 映射：look 与校正分离；产品/服装场景应有「品牌色锁定」，与 Gore 的 context of nature 是同一类锁，只是锁的对象从草木变成织物。

**Chris Burkard 及同类户外商业**

- 同样：现场光方向保留、对比短、主体小、 palettes 跟气候走，而不是跟 LUT 走。

### 3.5 婚礼 / 人像胶片模拟

**Mastin Labs 3-Step**（Kirk Mastin）

1. 选对胶片预设（Fuji 400H 偏轻盈；Portra pushed 偏暗情绪）。
2. 拉曝光。胶片高光肩部会「吞」过曝，所以套预设后往往要**加**曝光，而不是减。
3. 白平衡 / 色调。

附加：扫描的 All Soft / All Hard / Highlight Soft 等，是胶片扫描时的高光/阴影肩，不是 HSL。
按肤色与天气选片种：浅肤/阴天偏 Fuji 400H；深肤/晴天偏 Portra 400。

**VSCO Film X / Archetype Process / Dehancer / FilmConvert**

- 胶片 = 色调 + 颗粒结构 + 高光滚降 + （可选）halation，不是一条 S 曲 + 加点 grain。
- 清晰度常要略降，才能像扫描底片；用 texture 补中频，而不是再加 clarity。

对 SuperPhoto 的硬含义：`filmic_portrait_warm` 现在是温度+HSL+split tone 的加和。若要更像 Portra，缺一条 **highlight rolloff 曲线**（`ToneCurve` 已存在，可承担），而不是再加 saturation。

### 3.6 人像 / 美妆精修

标准顺序（Pratik Naik, Julia Kuzmenko McKim, Retouching Academy）几乎锁死：

```
RAW 一次校正（皮肤曝光、白平衡、不要在 RAW 里磨皮）
→ 污点/愈合（只去暂时性缺陷：痘、血丝、尘）
→ 纠正性 dodge & burn（匀斑驳色阶，2% flow，50–70% 缩放，看大不看毛孔）
→ 塑形 dodge & burn（颧骨、下颌、眼窝，单独一步）
→ 颜色统一（去红区、统一颈部与面颊）
→ 频率分离（可选，最后，小半径）
→ 眼/牙/发 局部
→ 输出锐化
```

禁令：

- 不要先模糊再「找回来」纹理。
- 不要在 200% 修。
- 痣、皱纹、肤色特征默认保留，除非客户/体裁要求（商业美妆 vs 纪实肖像是两个制度）。
- Liquify 属于几何/内容改变，必须受伦理制度约束。

**Lindsay Adler**：Capture One 做色彩与皮肤，Photoshop 做精修；后期强度随体裁变（肖像 < 美妆 < 前卫）。灯光解决的问题不要丢给磨皮。

**Peter Hurley 头像**：光比与表情是主菜；后期是匀皮肤、压衣服褶皱、眼白轻度、背景干净。头像 look 应接近 faithful + 轻微 subject lift，而不是电影青橙。

**Capture One Skin Tone / Color Editor**

- 先白平衡、先曝光，再进 Color Editor。
- Skin Tone 选项卡：把皮肤选区的色相/饱和/明度收束到一条走廊，而不是「把皮肤加橙」。
- Color Balance 三路：阴影/中间调/高光分离，可放在图层上（主体暖、背景冷）。
- 映射：现有 `portra_skin` 降橙红饱和、略提橙明度，方向对；缺「把皮肤色相方差收紧」的算子。

### 3.7 电影调色（给静照用的部分）

**Cullen Kelly**

- 客观级（曝光、白场、色温）和 look dev 是两种技能。
- 场景线性里曝光像停，显示空间里 offset 会把影子砸进地板。
- 先把主体放到正确曝光，再把「其余」压下去——不要把主体从欠曝里捞。
- Look 的一半是你**不**加的东西。
- 节点/LUT 要能按项目调制，不能当死滤镜。

对静照：`cinematic_teal_orange` 已经是 look 层。它必须叠在一次校正之后，且受皮肤/植被色相锁限制（青橙最容易把皮肤染成橘、把树染成棕）。现有 `_fit_lut_to_scene` 对部分 XMP 做了皮肤/品红保护，应推广到所有强 look。

### 3.8 商业时尚 / 通用 Lightroom 教学

**Scott Kelby 7-Point System**（2021 更新版）

1. 相机配置文件（Adobe Landscape / Color / Neutral 当不同起点）。
2. 白平衡。
3. 曝光 + 白场/黑场（Alt 拖到刚出现裁切）。
4. 对比。
5. 局部（蒙版）。
6. 颜色（HSL / 颜色分级）。
7. 收尾（效果、锐化、噪点）。

「先做该先做的」：裁切、镜头、去尘、去噪若需要，插在点 1 之前。这与 Glyn Dewis 的 Prepare 以及 SuperPhoto Axis A 一致。

**Andy Hutchinson「任何 RAW 软件的五段」**

Prep（demosaic / 镜头 / 去噪 / 配置文件 / 裁切）→ Light（WB、曝光、对比、高光阴影、白黑、曲线）→ Structure（texture / clarity / dehaze）→ Color → Finish。
这是把 Kelby 和 scene-referred 拼在一起的当代通用顺序。

**Glyn Dewis Prepare / Repair / Finesse**

专为 Lightroom AI 工具互相踩踏而写：先像素级重建，再内容修复，最后创意。SuperPhoto 若引入生成式去物，必须放在 Axis A，且纪实制度下关闭。

### 3.9 开源 scene-referred（darktable）

默认 sigmoid 或 filmic **二选一**，禁止叠 base curve。

- 先用 exposure 把中灰放到观感正确（曲线绕中灰旋转）。
- color calibration = 白平衡。
- color balance rgb = 颜色（放在显示变换之前）。
- tone equalizer = 按亮度分区的曝光（亮度蒙版的 scene-referred 版）。
- 显示变换只做动态范围压缩和肩部。

映射：SuperPhoto 的线性工作缓冲已经是 scene-referred 的入口；缺的是 tone equalizer（luma 分桶曝光）和唯一显示变换。

### 3.10 黑白美术（结构最清楚的「主体/背景」模型）

**Joel Tjintjelaar：Figure–Contrast–Depth / iSGM**

1. 标出 Figure（主体）和 Ground（其余一切）。
2. 硬蒙版管形状，亮度蒙版管光。
3. **先降低 Ground 的对比，再决定把它整体压暗或提亮**——否则你一压暗，背景对比反而更高。
4. 深度来自每个平面上的渐变（暗↔亮），不是全局 S 曲。
5. 高光用窄亮度选区单独保护。

映射：现有 `background_darken` / `background_desat` / `subject_lift` 是这个模型的简化。缺「先降背景对比」这一步，所以有的图背景压了以后更抢戏。

**Ansel Adams Zone System → 数字**

- 胶片：为阴影曝光，为高光显影。
- 数字：高光是硬切，所以保护高光，阴影在 RAW 里提。彩色不要落到 Zone 2 以下（浊）或 Zone 7 以上（褪）。
- 闪避加深是印相的一部分，不是作弊。
- 映射：`highlight_clip` / `shadow_p05` / `highlight_p95` 已在 `TechnicalReport`。可按体裁把目标 Zone 写成一次校正的约束，而不是「平均亮度 0.5」。

### 3.11 纪实 / 新闻伦理（制度，不是风格）

这些不是 look，是**允许集**。SuperPhoto 应按 job 或 genre 挂制度。

**NPPA / Writing With Light / Magnum**

- 允许：适度影调、颜色校正、裁切、去传感器尘。
- 禁止：增删物体、改变陈述、把灰天变蓝天、用闪避加深掩盖/凭空造出内容、生成式填充。
- Magnum 2024：合成影像不进档案库。

**World Press Photo 2026**

- 单帧、单次曝光。
- 色相不得显著偏离；密度/对比/饱和不得掩盖信息。
- 禁止生成式填充、禁止 Super Resolution / Topaz 类「制造新像素」放大。
- Denoise、自动色阶、物体选区做局部调整：有限度允许，以「不增删相机记录的信息」为界。
- 肖像同样不许去掉皮肤上的生理痕迹。

工程含义：`defect_heal`、`clutter_suppress`、几何重裁、强 LUT、语义补全，在 `documentary` 制度下应全部冻结。现有 Axis A 的 defect_heal 即使色中性，在 WPP 下也只允许传感器尘。

### 3.12 野生动物

Anthony Morganti / DxO PureRAW 6 / Funnel Wildlife Toolkit / Matt Kloskowski：

1. 未裁切的 RAW 上去噪（羽毛需要 XD 级细节引擎，人像则避免 XD 在平滑区造假纹理）。
2. 再裁切；必要时超分（注意：WPP 禁这一步，野生动物商业用途才用）。
3. 锐化打在眼睛、羽毛、毛发；背景不锐。
4. 背景降低存在感（暗一点、少一点饱和），眼睛从眼窝阴影里拿出来。
5. 颜色按时段配置文件（晨/午/阴/暮），不是一条 LUT 打天下。

映射：restoration 的 denoise 应在几何之前（现有 pipeline 是 restore 在 geometry 之前，对）。retouch 的 subject 组应对「眼睛」有更强权重；wildlife 不该套 `cinematic_teal_orange`。

### 3.13 电商 / 产品

与 look 体系几乎正交：

- 校正永远做：白平衡用色卡、曝光、对比、裁切。
- 抠图（clipping path）与背景规范（纯白、阴影分层）是交付，不是风格。
- 颜色准确性是第一验收：错色比「不好看」贵得多（退货）。
- 按产品家族同步，不按单张玩 look。
- 导出 sRGB。服装品牌色锁定（与 Strohl 商业片相同）。

映射：`product` genre 的默认 look 应是 faithful；HSL 不得把商品主色移出 ΔE 门槛。现有 `genre_tags` 已有 product，但 `editorial_crisp` 可能被误配。

---

## 4. 冲突表（引擎必须显式选择，不能平均）

| 冲突 | 甲方 | 乙方 | SuperPhoto 建议 |
|---|---|---|---|
| 先 look 还是先曝光 | Mastin：先胶片特性 | Kelby / C1 / Cullen：先 primary | look 若 `changes_highlight_rolloff`，先上 tone-map 再微调 EV；否则校正先行 |
| 全局先还是局部先 | 多数 LR 教程：全局再局部 | 部分风光（Lenscraft 等）：先局部再全局，防过修 | 一次校正全局；对比与气氛走局部。禁止「全局 clarity+dehaze+sat」 |
| 阴影该不该提亮 | 数码默认提阴影 | Heaton / 低键 / 纪实：主体以外可以很暗 | 由 genre + highlight_clip 决定；有天空裁切时禁止为提阴影而加曝光 |
| 绿要不要饱和 | 旅游片要「翠」 | Gore / 陈建强：绿要静、要橄榄 | 默认走 Gore 走廊；look 里显式 opt-in 才能加绿饱和 |
| 磨皮 | 美妆商业要干净 | 纪实/WPP 禁止改生理痕迹；Naik：保留纹理 | ethics regime 开关；默认 D&B 匀色阶，不做低频模糊 |
| 去杂物 | 商业/风光常做 | NPPA/WPP 禁止 | `clutter_suppress` / `defect_heal` 按制度冻结 |
| 清晰度 | editorial_snap 靠 clarity | retouch 轴禁止锐化/clarity | 杂志 look 应改走「中灰对比 + 局部 texture」，或把 texture 做成修复轴之外的第三条「结构」配额 |
| HDR 合成 | Locardi / Bagshaw / Adamus 在极端动态范围下做 | Benn / Popsys：单张 | v0 无包围曝光输入，保持单张；用分区 luma 代替 HDR |
| 生成式修补 | Lightroom Remove、Glyn Repair | WPP 直接 DQ | 默认关；仅 commercial 制度可开，且写入 sidecar |

---

## 5. 原语清单（ impl 时只实现这些，不要实现摄影师）

把上面所有流程压成一组稳定算子。名字对应建议的代码边界，不一定要一次全做。

### 5.1 制度 `EthicsRegime`

```
documentary | commercial_portrait | landscape_expressive | product | wildlife
```

| 算子 | documentary | portrait | landscape | product | wildlife |
|---|---|---|---|---|---|
| restoration denoise/sharpen | 有限度 | 开 | 开 | 开 | 开（裁前） |
| defect_heal | 仅传感器尘 | 暂时性瑕疵 | 尘/热像素 | 表面尘 | 尘 |
| clutter_suppress | 关 | 弱 | 开 | 背景规范 | 弱 |
| geometry crop | 仅裁多余 | 开 | 开 | 按模板 | 开 |
| 强 LUT / 青橙 | 关 | 皮肤锁下可 | 自然色锁下可 | 关 | 关 |
| 生成式去物 | 关 | 可选 | 可选 | 关（改用抠图） | 关 |

### 5.2 一次校正 `PrimaryGrade`（客观、可单独 QA）

字段已在 `CorrectionHint`：EV、色温、色调、高光、阴影、白场、黑场、轻微对比。

新增约束，来自 Kelby / 数字 Zone / Heaton：

- 白场/黑场用分位而不是均值：贴近 `highlight_p95` / `shadow_p05`，停在裁切出现之前。
- 有 `highlight_clip > 0` 时，禁止正 EV（现有 critique 已有类似 `do_not`）。
- 人像：皮肤均值落到 Zone 5–6（约 0.45–0.65 显示亮度），不是整图均值 0.5。
- 产品：灰卡/白底优先于「好看的对比」。

### 5.3 亮度分区 `LumaBands`

来自 TK / Bagshaw / darktable tone equalizer / Lightroom luminance range。

建议 5 桶：deep_shadow / shadow / mid / highlight / specular。
任何局部曝光、去雾、饱和，默认可选「乘以某桶」。
Heaton 的「高光径向只打在最亮处」、Locardi 的「曲线只动天空高光」，都是这一原语。

### 5.4 语义分区（已有）

`LayerMasks`：subject, sky, foliage, water, skin, background, clutter, defects。
继续作为「是什么」；`LumaBands` 作为「有多亮」。二者相交才是风光/人像精修的常用选区（天空∩高光、皮肤∩中间调）。

### 5.5 色相走廊锁 `HueGuard`

来自 Gore context of nature、陈建强、Strohl 品牌色、C1 Skin Tone、WPP 色相规则。

| 类 | 允许色相位移 | 饱和 | 明度 |
|---|---|---|---|
| 皮肤 | 极小（橙红走廊内） | 只降不猛加 | 可微提 |
| 植被绿 | ±8–12° 内（黄绿↔蓝绿） | 默认降 | 可微降 |
| 天空蓝 | 禁止变青/品红 | 可微加 | 可降（更深的天） |
| 产品主色 | ΔE 门槛 | 几乎锁 | 锁 |
| 直射光区 | 较大 | 允许更高 | 允许更高 |
| 无光区 | 几乎锁 | 强制低 | 保持暗 |

实现上：look 的 HSL 在合并后，按像素类别 clamp，而不是按滑杆 clamp。

### 5.6 闪避加深 `DodgeBurn`

来自 Adams / Page / Naik / Kuzmenko / Percy。

两层意图必须分开（人像课里已经分开）：

- **纠正性**：降低空间色阶方差（斑驳皮肤、斑驳墙），不改变体积。
- **塑形**：沿已有光方向加强体积与视觉路径（Adamus 的过渡向量）。

实现建议：低 flow 的局部 EV，蒙版 = 语义 ∩ 亮度；禁止变成全局 contrast。
背景侧优先「先降对比再降亮度」（Tjintjelaar），对应新字段例如 `background_contrast` 或在 `background_darken` 前对背景做一次 contrast 负向。

### 5.7 显示变换 `DisplayTransform`

来自 darktable sigmoid/filmic、Cullen look LUT、Mastin 胶片肩部、`Recipe.curve`。

一条曲线（或 LUT）负责：中灰对比、高光滚降、黑场肩。
Look 的「胶片感」应主要活在这里，而不是 12 个滑杆各加一点。
`lut_mix` 已经是混合系数，应对这条变换做 mix，而不是对全部滑杆做 mix。

### 5.8 结构配额 `Structure`（建议从 retouch 里独立，或成为第 4 意图）

Texture / 中频对比 / 去雾 被 Hutchinson 单列为 Structure，被 SuperPhoto 从 retouch 剥掉。

建议：

- restoration 继续只做去噪与捕获锐化。
- 新增受制度控制的 `texture`（中频，天空权重低）。
- `dehaze` 只允许 sky ∩ highlight，数值封顶（景观教程经验：>+0.25 开始假）。
- editorial look 用 texture + 中灰对比，不恢复全局 clarity。

### 5.9 验收 `Gates`

在现有 `score_grade` / `score_restoration` 上加锁，而不是再打一个分数：

| 锁 | 来源 | 失败时 |
|---|---|---|
| 修复色漂/整体明暗 | 现有 restoration gate | 丢弃 Axis A |
| 高光新裁切 | Heaton / Zone / QA | 回退 EV/whites |
| 植被/天空/皮肤色相越走廊 | Gore / WPP / C1 | 回退 HSL/grade |
| 皮肤高频能量下降过多 | Naik / Kuzmenko | 回退任何平滑 |
| 绿饱和暴涨 | nuclear greens | 回退 sat/vibrance |
| 背景对比不降反升 | Tjintjelaar | 回退 background 操作 |
| 内容增删 | WPP/NPPA | 制度直接禁止该算子 |
| 存在感过低且 slot≠A | 现有 too_weak | 轻微加强 look mix，永不加强 restoration |

「通透」可写成 finish 子分：中灰 std 上升、高光未裁、饱和增益 < 阈值、阴影仍有分离（不是一块死黑）。

---

## 6. 与当前 SuperPhoto 的对照

### 6.1 已经对齐的

| 共识 | 代码位置 |
|---|---|
| 修复 / 美化分轴 | `workflow.py`，`pipeline.py` restore→retouch |
| 修复色中性 | `apply_restoration` 拆 luma/chroma 再粘回 |
| 美化最多三个意图 | `split_retouch_recipe` |
| 美化不做去噪锐化 | 强制清零 clarity / subject_sharpen / defect_heal |
| 校正与 look 分离后合并 | `preference.build_plans` |
| Heaton 骨架、Gore 绿、青橙、Portra 皮 | `recipes.MOVES` |
| 安全分 > 存在感 | `qa.score_grade` 0.58 / 0.20 / 0.22 |
| 语义分区 | `layers.py` + `semantic.py` |
| 场景线性工作缓冲 | `ingest.load_linear` |
| look 随场景收敛 | `_fit_lut_to_scene`、`_adapt_to_scene` |
| 用户选择反哺 look | `preference.preferred_look_ids` |

### 6.2 缺口（按对画质的杠杆排序）

1. **没有伦理制度**，只有 genre。纪实片与商业片共用 `clutter_suppress` / `defect_heal` / 强 LUT，会在错误的一边犯规。
2. **没有 luma 分桶**。风光深度（Bagshaw/Kuyper/Heaton 亮度范围）全靠语义近似，天空里的暗云和亮云被同样处理。
3. **Dodge & burn 不是一等原语**。现在只有 `subject_lift` 与 `background_darken`，没有「沿光方向塑形」和「纠正性匀色阶」。
4. **Look 是滑杆加和，不是显示变换**。Mastin/filmic/sigmoid 的高光肩无法表达；`ToneCurve` 存在但未成为 look 的主载体。
5. **校正与 look 合并后再拆意图**，一次校正可能被 3-attribute 配额吃掉（例如 look 占用 color+atmosphere，校正里的 tone 还在，但 skin HSL 可能被丢）。primary 应豁免配额。
6. **HueGuard 未产品化**。`gore_greens` 与 `_fit_lut_to_scene` 是特判，不是通用锁。青橙对人像皮肤、Velvia 对植被都可能越界。
7. **Tjintjelaar 的「先降背景对比」不存在**，背景压暗后有时更抢。
8. **editorial_snap 依赖的 clarity 在 retouch 被清零**，杂志槽位名存实亡。
9. **饱和度未乘亮度**。全局 vibrance 仍会给暗部染色，违反 Gore/陈建强。
10. **人像缺皮肤色相方差收敛**（C1 Skin Tone），只有橙红 HSL。
11. **产品/野生动物缺少专用默认 look 与冻结表**。
12. **Restoration 不做镜头校正 / CA**（Kelby「先做该先做的」、Heaton 第一步）。JPEG 输入尤其明显。
13. **Critique 的 referred pain 还不够**：有 issues 和 do_not，但没有「不要提主体、去压背景」这种因果改写规则。
14. **没有输出媒介锐化阶段**（屏预览 vs 导出）。现在预览 JPEG 的锐化与修复锐化混在一起。

### 6.3 建议的增量（不在本文实现，只标定方向）

保持两轴，不要变成「又一个全能 PS」。按杠杆分批：

**P0 — 把已有轴用对**

- 给 job 加 `ethics_regime`（可由 genre 推断，可被覆盖）。
- Primary 豁免 3-attribute 配额：拆 recipe 时先留下校正，再对 look 限额。
- HueGuard 作为 QA 锁 + 合并后 clamp。
- 饱和/vibrance × luma 权重。

**P1 — 风光与人像都会立刻变好**

- `LumaBands` 交语义蒙版。
- `background` 先降对比再暗。
- DodgeBurn 纠正性（皮肤/墙）走低幅度局部 EV。
- `filmic_portrait_warm` / 胶片 XMP 把主对比迁到 `ToneCurve` 肩部。

**P2 — 结构与制度**

- `texture` 作为可选第 4 意图或 Structure 轴，供 editorial 使用。
- documentary 冻结表。
- 产品 ΔE 锁。
- 镜头/CA 进入 Axis A（仍须色中性）。

---

## 7. 体裁 → 默认意图（给 `compose_look_ids` / split 用）

在「最多三个美化方向」里，体裁决定哪三个，而不是让强度最大的三个胜出。

| 体裁 | 默认三意图 | 默认 look 家族 | 冻结 |
|---|---|---|---|
| landscape | tone, color, background | faithful / landscape_filmic | 全局 sat；绿越走廊 |
| portrait | tone, color, subject | faithful / filmic_portrait_warm | 皮肤模糊；青橙不锁皮 |
| street / documentary | tone, color, （弱）atmosphere | faithful / moody_lowkey | heal/clutter/强 LUT |
| night | tone, atmosphere, subject | moody_lowkey | 强提阴影导致噪点回翻（应回 Axis A 而不是加 EV） |
| product | tone, color, background | faithful / editorial_crisp | 任何主色位移 |
| wildlife | tone, subject, background | faithful / landscape_filmic | 背景锐化；裁后去噪 |
| fashion editorial | tone, color, （structure/texture） | editorial_crisp | 塑料皮 |
| wedding film | color, tone, atmosphere | filmic_portrait_warm + 胶片曲线 | 先破坏高光肩再套 look |

Slot A 继续当「能交出去的克制版」（Heaton/faithful）；B/C 才允许更强 look。这与现有 `DEFAULT_LOOK_IDS` 和 QA 对 A 更宽容的 `too_weak` 规则一致。

---

## 8. 来源（公开、可核验）

只列对本文有直接句子或步骤贡献的入口。付费课程只引用其公开大纲/访谈里出现过的方法论，不引用盗版材料。

**风光**

- Thomas Heaton, *How I Edit My Images* 及 *Image Processing for Landscape Photography* 播放列表；Adobe Live with Joe Allam。
- Ted Gore, *Amongst Giants*（Outdoor Photographer）；*Color Theory for Landscape Photography*；Fstoppers *The Real Versus the Beautiful*。
- Sean Bagshaw, Outdoor Exposure / TK；PhotoPills 2h19m 直播；Visual Wilderness 工作流文。
- Tony Kuyper, luminosity masks 系列（2006 起）。
- Nick Page, *Mastering Dodging and Burning*。
- Marc Adamus, Outdoor Photographer *Next Generation Landscapes*；1x 访谈；clone-paint / history-paint 讲座。
- Daniel Kordan, Lightroom Masterclass 大纲。
- Elia Locardi, *Photographing the World* 系列（手工曝光混合，反 HDR 味）。
- Alister Benn, Expressive Photography；*Understanding the Landscape*；CaptureLandscapes POTY 访谈。
- Bruce Percy, 博客：simplify、tonal separation、pendulum of colour、human eye / grads。
- Greg Benz, Lumenzia；ACR Glow / 工作流笔记。
- 陈建强，《摄影金奖之路：风光摄影创作与后期核心技法》。
- 「Thomas 看看世界」，《风光摄影后期基础》（电子工业出版社）：局部是核心、六大合成、三步锐化。

**人像 / 美妆 / 婚礼**

- Pratik Naik, Portrait Masters *Retouching Guide*；CreativeLive *Portrait Retouching Basics*。
- Julia Kuzmenko McKim, Retouching Academy / Beauty Retouch Panel；Professional Beauty Retouching 课程大纲。
- Lindsay Adler, Learn+ EP57 Retouching；CreativeLive 精修课。
- Capture One, *Working with Colour*；Skin Tone / Color Editor 官方直播。
- Mastin Labs, *The Mastin Labs 3-Step Workflow*。
- VSCO, Film X 方法说明。

**调色 / 通用 RAW**

- Cullen Kelly, Grade School（look vs grading；scene vs display；referred pain）。
- Scott Kelby, *Lightroom 7-Point System*（2021）及 KelbyOne 课大纲。
- Andy Hutchinson, *One Universal Workflow for Literally Any RAW Photo Editor*。
- Glyn Dewis, Lightroom *Prepare, Repair, Finesse*。
- darktable 手册：sigmoid / filmic rgb；scene-referred 默认工作流。
- DxO PureRAW 6：去噪在创意之前；羽毛/毛发用 XD 引擎。

**伦理**

- NPPA Code of Ethics；Writing With Light。
- Magnum Photos Code of Ethics；2024 合成影像声明。
- World Press Photo, *What counts as manipulation*；2026 Entry Rules（含 AI 条款）。

**其他体裁**

- Anthony Morganti, *Post Processing Feathers & Fur*。
- Matt Kloskowski × Topaz, wildlife 顺序。
- 电商：产品后期校正 / clipping / 色准工作流（家族同步、sRGB、色卡）。
- Joel Tjintjelaar, Figure–Contrast–Depth；iSGM。
- Ansel Adams, *The Negative* / *The Print*；数字 Zone 的当代转述（Outdoor Photographer 等）。
- Alex Strohl, AS1/AS2 预设说明与 *edit like a painter*。

**已写进代码注释、本文与之对齐的短名单**

- `recipes.py` 已记载：Heaton（高光→对比，不爆饱和）、Gore（绿要静，饱和跟着光）、调色师（暖光冷影）、人像（皮肤优先，阴影大部分保留）。

---

## 9. 一句话给实现者

顶尖摄影师公开的后期，收束以后不是「更多滑杆」，而是：

**在一种伦理制度下，先做色中性修复，再做可回滚的一次校正，再用亮度×语义的局部塑光和色相走廊来写关系，最后才混入一条显示变换式的 look；安全锁优先于味道。**

SuperPhoto 的两轴、三意图、安全分已经踩在这条路上。下一步不是再堆 XMP，而是把 luma 分桶、HueGuard、primary 豁免配额、以及制度冻结表补齐——让 Heaton、Gore、Naik、Kelly、WPP 和陈建强可以共享同一根管道，只换政策和配方。
