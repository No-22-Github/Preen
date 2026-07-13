macOS 应用图标包
==================

包含内容:
1. AppIcon.icns          - 可直接用的 macOS 图标文件，拖进 Xcode 项目或替换 App 的图标资源即可
2. icon.iconset/         - Apple 官方 iconset 文件夹（10 个标准尺寸 PNG），用于：
                            - Xcode Assets.xcassets 里手动拖入各尺寸
                            - 或在 Mac 上执行以下命令重新生成 .icns：
                              iconutil -c icns icon.iconset
3. logo_source.svg       - 矢量源文件，无损缩放，方便以后改色/改尺寸/重新导出

尺寸清单 (icon.iconset):
  icon_16x16.png        16x16      (1x)
  icon_16x16@2x.png      32x32      (2x)
  icon_32x32.png         32x32      (1x)
  icon_32x32@2x.png      64x64      (2x)
  icon_128x128.png       128x128    (1x)
  icon_128x128@2x.png    256x256    (2x)
  icon_256x256.png       256x256    (1x)
  icon_256x256@2x.png    512x512    (2x)
  icon_512x512.png       512x512    (1x)
  icon_512x512@2x.png    1024x1024  (2x)

设计说明:
  已按 macOS 图标规范做了安全边距处理（内容占画布约 78%，四周留白），
  背景为透明。如果实际放进 Dock 感觉偏小/偏大，可以告诉我调整留白比例重新导出。
