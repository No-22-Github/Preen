# Preen 品牌素材

| 素材 | 用途 |
|---|---|
| `../preen_title.png` | GitHub / README 头图。保留白色背景，避免网页主题影响可读性 |
| `macos/Preen/Assets.xcassets/PreenTitle.imageset` | App 首页标题图。浅色外观用黑色透明图，深色外观用白色透明图 |
| `preen_app_icon_source.svg` | App 图标矢量源文件，透明外圈 + 白色圆角底板 + 黑色线稿 |
| `macos/Preen/Assets.xcassets/AppIcon.appiconset` | Xcode 使用的 10 个 macOS 标准尺寸 PNG |

规则：GitHub 不使用透明标题图；App 不使用带白底的 GitHub 头图。界面引用 Asset Catalog 名称 `PreenTitle`，不要直接写文件路径。

AppIcon 在透明画布内使用带安全边距的白色圆角底板，不能把整张方形画布铺白，否则 Dock 会显示内层硬方块。标题图仍保留透明背景，由 Asset Catalog 按深浅外观切换黑白版本。
