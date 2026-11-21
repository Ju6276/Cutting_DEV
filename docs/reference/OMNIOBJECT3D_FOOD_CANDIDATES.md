# OmniObject3D food candidates

2026-11-21: 50 non-packaged candidate mesh IDs across 12 food categories, plus 7 excluded packaging/stick records (57 models total). These are a discovery shortlist, not calibrated materials or accepted simulation assets.

Official project: https://omniobject3d.github.io/ . Converted mirror: https://huggingface.co/datasets/AlbertS666/OmniObject3D-objects , pinned revision c94996f67d9fb212b10df36e938a49602779a247. Verified both Object.usd and textures/Scan.jpg in the zip central directory for every ID below. The initial filename audit used archive metadata only. The follow-up gallery fetched all 57 USD models and 57 textures by range requests, without downloading the complete archive. All previews were visually screened for packaging/preparation. This does not establish watertightness, physical scale or simulation acceptance.

- cucumber (黄瓜): `cucumber_001`, `cucumber_002`, `cucumber_003`, `cucumber_004`, `cucumber_005`, `cucumber_006`. 扫描包含黄瓜段/纵切片，不能一概称整条；比较几何和种子区；扫描表皮不是内部结构。
- potato (土豆): `potato_001`, `potato_002`, `potato_003`, `potato_004`, `potato_005`, `potato_006`. 先制备半颗、平面朝下；扫描整颗不能冒充半颗。
- carrot (胡萝卜): `carrot_001`, `carrot_002`, `carrot_003`, `carrot_004`, `carrot_005`, `carrot_006`. 适合较硬根茎；记录新鲜程度与纤维方向。
- sweet_potato (红薯): `sweet_potato_001`, `sweet_potato_002`, `sweet_potato_003`, `sweet_potato_004`. 块根；控制尺寸和生熟状态。
- yam (山药): `yam_001`, `yam_002`, `yam_003`. 根茎；核对资产去皮状态与实物一致。
- apple (苹果): `apple_001`, `apple_002`, `apple_003`, `apple_005`. 去核切块或避开果核；皮与肉不能直接视为同材质。
- pear (梨): `pear_001`, `pear_002`, `pear_003`, `pear_004`. 避开硬核；成熟度和含水量需要记录。
- mushroom (蘑菇): `mushroom_001`, `mushroom_002`, `mushroom_003`, `mushroom_004`, `mushroom_005`. 选单体；菌柄和菌盖不同，不默认均匀材料。
- cheese (奶酪): `cheese_001`, `cheese_003`, `cheese_004`, `cheese_005`. 图库实查为包装/带棒柄，暂排除，不能直接用作裸食材。
- sausage (香肠): `sausage_010`, `sausage_011`, `sausage_012`. 图库实查有包装标签，暂排除，不能直接用作裸食材。
- banana (香蕉): `banana_001`, `banana_002`, `banana_003`. 这三个扫描均带皮，不能替代当前去皮香蕉资产。
- bread (面包): `bread_001`, `bread_002`, `bread_003`. 多孔和外壳；当前均匀模型只是近似。
- cake (蛋糕): `cake_001`, `cake_002`, `cake_003`. 优先无装饰单层；奶油/夹层不能共用一个材料值。
- rice_cake (年糕): `rice_cake_001`, `rice_cake_002`, `rice_cake_003`. 黏弹性和黏附明显；需要额外模型核对。

Start with a small set of real food preparations, then use several shape instances per category. Texture and mesh shape do not supply Young's modulus, friction, density or fracture parameters. Match actual food preparation, temperature, maturity and dimensions when measuring with UR10e/ATI Mini45. Banana scans may include peel and must not replace the current peeled homogeneous banana silently. The mirror has no tofu/luncheon category by those names; current assets remain the existing prepared geometries. Physically heterogeneous, porous and viscoelastic foods require additional modelling checks.

Full machine-readable checklist and CSV: /dataHDD/dong/Research/food_meshes/omniobject3d_shortlist_20261121/shortlist_50.{json,csv}. No downloads or data conversions should silently enter the training distribution.

Offline HTML gallery (model views, original texture atlases, exclusion reasons,
and current five-food Nano capture images):
/Users/judong/Documents/Cutting_DEV/Reports/OmniObject3D_20261121/gallery/index.html .
It opens directly in a browser without Canvas, npm or a web server.
Model source files stay in the remote research asset directory; local delivery
contains previews and provenance. Banana skins, porous bread/cake and preparation
differences remain modelling decisions, not approved additions to training.
