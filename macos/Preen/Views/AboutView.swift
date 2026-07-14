//
//  AboutView.swift
//  Preen
//
//  关于 Preen 独立窗口:app icon + 名称 + 版本 + 描述 + GitHub / 引用项目致谢墙。
//  由状态底栏右侧 info 图标 / app 菜单「关于 Preen」触发(openWindow)。
//

import SwiftUI
import AppKit

struct AboutView: View {
    /// 版本号从 Bundle 读取(构建时由 pbxproj MARKETING_VERSION 注入)。
    private var versionText: String {
        let v = Bundle.main.infoDictionary?["CFBundleShortVersionString"] as? String ?? "1.0"
        let build = Bundle.main.infoDictionary?["CFBundleVersion"] as? String ?? "1"
        return "版本 \(v) (\(build))"
    }

    /// 双列网格列定义。
    private let creditColumns: [GridItem] = [
        GridItem(.flexible(), spacing: 6),
        GridItem(.flexible(), spacing: 6)
    ]

    /// 引用项目清单(名称 / 简述 / 许可证 / 仓库链接)。许可证经 GitHub/HF API 核实。
    private let credits: [Credit] = [
        .init(name: "MLX",
              blurb: "Apple 机器学习框架，张量运算与自动微分的底层地基",
              license: "MIT",
              icon: "cpu",
              url: "https://github.com/ml-explore/mlx"),
        .init(name: "MLX-LM",
              blurb: "核心训练/推理引擎，提供 rwkv7.py 前向与自动微分",
              license: "MIT",
              icon: "gearshape.2",
              url: "https://github.com/ml-explore/mlx-lm"),
        .init(name: "Flash Linear Attention",
              blurb: "线性注意力上游库，模型转换校验基准",
              license: "MIT",
              icon: "bolt",
              url: "https://github.com/fla-org/flash-linear-attention"),
        .init(name: "RWKV-PEFT",
              blurb: "RWKV 参数高效微调方法参考",
              license: "Apache-2.0",
              icon: "wrench.and.screwdriver",
              url: "https://github.com/Joluck/RWKV-PEFT"),
        .init(name: "RWKV-LM",
              blurb: "BlinkDL 维护的 RWKV 模型仓库，提供原始权重与参考实现",
              license: "Apache-2.0",
              icon: "books.vertical",
              url: "https://github.com/BlinkDL/RWKV-LM"),
        .init(name: "RWKV Runner",
              blurb: "导出的 .pth 初始 state 挂载目标，与 RWKV 生态直连",
              license: "MIT",
              icon: "arrow.down.doc",
              url: "https://github.com/josStorer/RWKV-Runner"),
        .init(name: "NekoQA-10K",
              blurb: "猫娘风格 QA 对话数据集，风格迁移训练数据来源",
              license: "Apache-2.0",
              icon: "person.crop.circle.badge.questionmark",
              url: "https://huggingface.co/datasets/liumindmind/NekoQA-10K"),
        .init(name: "rwkv7-0.1B-g1",
              blurb: "World Tokenizer 与转换校验模板来源",
              license: "Apache-2.0",
              icon: "textformat",
              url: "https://huggingface.co/fla-hub/rwkv7-0.1B-g1"),
        .init(name: "Transformers",
              blurb: "Hugging Face 模型库，转换链路的 HF 格式基准",
              license: "Apache-2.0",
              icon: "square.stack.3d.up",
              url: "https://github.com/huggingface/transformers"),
        .init(name: "safetensors",
              blurb: "转换产物张量格式（.pth → HF safetensors）",
              license: "Apache-2.0",
              icon: "shippingbox",
              url: "https://github.com/huggingface/safetensors"),
        .init(name: "swift-markdown-ui",
              blurb: "App 内 Markdown 渲染（消息与文档）",
              license: "MIT",
              icon: "doc.richtext",
              url: "https://github.com/gonzalezreal/swift-markdown-ui"),
        .init(name: "uv",
              blurb: "构建工具链与依赖管理",
              license: "Apache-2.0",
              icon: "hammer",
              url: "https://docs.astral.sh/uv/")
    ]

    var body: some View {
        VStack(spacing: 20) {
            header

            // 引用项目致谢:双列网格,可滚动。
            VStack(alignment: .leading, spacing: 10) {
                Text("引用项目与致谢")
                    .font(.subheadline.weight(.semibold))
                    .foregroundStyle(.secondary)

                ScrollView(.vertical, showsIndicators: false) {
                    LazyVGrid(columns: creditColumns, spacing: 6) {
                        ForEach(credits) { credit in
                            CreditCard(credit: credit) { open(credit.url) }
                        }
                    }
                }
            }
            .frame(maxWidth: .infinity, alignment: .leading)

            Text("许可证信息以各项目仓库为准")
                .font(.caption2)
                .foregroundStyle(.tertiary)
        }
        .padding(28)
        .frame(width: 560, height: 560)
    }

    // MARK: - 头部

    private var header: some View {
        VStack(spacing: 8) {
            Image(nsImage: NSApplication.shared.applicationIconImage)
                .resizable()
                .scaledToFit()
                .frame(width: 92, height: 92)

            Text("Preen")
                .font(.title.bold())

            HStack(spacing: 6) {
                Image("RWKVLogo")
                    .renderingMode(.template)
                    .resizable()
                    .scaledToFit()
                    .frame(width: 15, height: 15)

                Text("RWKV-7 State Tuning")
            }
            .font(.callout)
            .foregroundStyle(.secondary)

            Text(versionText)
                .font(.caption)
                .foregroundStyle(.tertiary)

            Button {
                open("https://github.com/No-22-Github/Preen")
            } label: {
                Label("GitHub 仓库", systemImage: "arrow.up.right")
                    .font(.callout)
            }
            .buttonStyle(.link)
            .padding(.top, 2)
        }
    }

    // MARK: - 辅助

    private func open(_ urlString: String) {
        if let url = URL(string: urlString) {
            NSWorkspace.shared.open(url)
        }
    }
}

// MARK: - 致谢卡片

/// 扁平可点卡片:默认透明,悬停时浮现淡背景。名称 + 简述 + 许可证内联。
private struct CreditCard: View {
    let credit: Credit
    let action: () -> Void

    @State private var hovering = false

    var body: some View {
        Button(action: action) {
            VStack(alignment: .leading, spacing: 5) {
                HStack(spacing: 7) {
                    Image(systemName: credit.icon)
                        .font(.callout)
                        .foregroundStyle(.secondary)
                        .frame(width: 18)

                    Text(credit.name)
                        .font(.callout.weight(.medium))
                        .foregroundStyle(.primary)
                        .lineLimit(1)

                    Spacer(minLength: 0)

                    Image(systemName: "arrow.up.right")
                        .font(.caption2)
                        .foregroundStyle(.tertiary)
                        .opacity(hovering ? 1 : 0)
                }

                Text(credit.blurb)
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .lineLimit(2)
                    .fixedSize(horizontal: false, vertical: true)
                    .frame(maxWidth: .infinity, alignment: .leading)

                Text(credit.license)
                    .font(.caption2)
                    .foregroundStyle(.tertiary)
            }
            .padding(.horizontal, 10)
            .padding(.vertical, 9)
            .frame(maxWidth: .infinity, alignment: .leading)
            .background(
                RoundedRectangle(cornerRadius: 8, style: .continuous)
                    .fill(Color.primary.opacity(hovering ? 0.06 : 0))
            )
            .contentShape(RoundedRectangle(cornerRadius: 8, style: .continuous))
        }
        .buttonStyle(.plain)
        .onHover { hovering = $0 }
    }
}

// MARK: - 数据模型

private struct Credit: Identifiable {
    let id = UUID()
    let name: String
    let blurb: String
    let license: String
    let icon: String
    let url: String
}
