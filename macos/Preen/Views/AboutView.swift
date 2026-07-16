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
    /// 渡鸦彩蛋:前 7 下(RWKV-7 的 7)是解锁门槛;解锁后每点一下切换一条文案。
    @State private var ravenTaps = 0
    @State private var lastTapAt: Date = .distantPast
    @State private var showEgg = false
    /// 当前要显示的彩蛋文案。第 0 条固定压轴,其余随机。
    @State private var eggMessage = ""
    /// 是否已解锁(首次点满 7 下后永久置真,之后每点切换文案)。
    @State private var eggUnlocked = false
    /// 上一次抽中的全局下标,用于避免解锁后连续两次抽到同一条。
    @State private var lastEggIndex = 0
    /// 气泡自动收起的计时器;连点换条时取消旧计时重排,防止提前关闭。
    @State private var dismissWork: DispatchWorkItem?

    /// 彩蛋文案池。索引 0 固定在第 7 下压轴出场,其余随机抽取。
    private static let eggMessages: [String] = [
        "Make RWKV Great Again",   // 固定压轴
        "RNN is All You Need",
        "There is no attention",
        "Just recur it",
        "Attention is overrated",
        "I'll be recurrent",
        "One State to rule them all",
        "State go brrr",
        "Got state?",
        "Fear the Goose",
    ]

    /// 版本号从 Bundle 读取(构建时由 pbxproj MARKETING_VERSION 注入)。
    private var versionText: String {
        let v = Bundle.main.infoDictionary?["CFBundleShortVersionString"] as? String ?? "1.0"
        let build = Bundle.main.infoDictionary?["CFBundleVersion"] as? String ?? "1"
        return L10n.format("版本 %@ (%@)", v, build)
    }

    /// 双列网格列定义。
    private let creditColumns: [GridItem] = [
        GridItem(.flexible(), spacing: 6),
        GridItem(.flexible(), spacing: 6)
    ]

    /// 引用项目清单(名称 / 简述 / 许可证 / 仓库链接)。许可证经 GitHub/HF API 核实。
    /// 排序口径:按对 Preen 的实际贡献从核心到外围——引擎地基 → 关键加速 → 模型权重 →
    /// 方法数据 → 导出校验链路 → 格式与基础设施。
    private let credits: [Credit] = [
        .init(name: "MLX-LM",
              blurb: "核心训练/推理引擎，提供 rwkv7.py 前向与自动微分",
              license: "MIT",
              icon: "gearshape.2",
              url: "https://github.com/ml-explore/mlx-lm"),
        .init(name: "MLX",
              blurb: "Apple 机器学习框架，张量运算与自动微分的底层地基",
              license: "MIT",
              icon: "cpu",
              url: "https://github.com/ml-explore/mlx"),
        .init(name: "rwkv-metal",
              blurb: "训练加速来源，WKV7 Metal checkpoint kernel 移植自此",
              license: "Apache-2.0",
              icon: "gauge.with.dots.needle.67percent",
              url: "https://github.com/RafaelUI/rwkv-metal"),
        .init(name: "RWKV-LM",
              blurb: "BlinkDL 维护的 RWKV 模型仓库，参考实现",
              license: "Apache-2.0",
              icon: "books.vertical",
              url: "https://github.com/BlinkDL/RWKV-LM"),
        .init(name: "rwkv7-g1",
              blurb: "RWKV-7 G1 官方权重，实际下载与转换的来源",
              license: "Apache-2.0",
              icon: "square.and.arrow.down",
              url: "https://huggingface.co/BlinkDL/rwkv7-g1"),
        .init(name: "RWKV-PEFT",
              blurb: "RWKV 参数高效微调方法参考",
              license: "Apache-2.0",
              icon: "wrench.and.screwdriver",
              url: "https://github.com/Joluck/RWKV-PEFT"),
        .init(name: "NekoQA-10K",
              blurb: "猫娘风格 QA 对话数据集，风格迁移训练数据来源",
              license: "Apache-2.0",
              icon: "person.crop.circle.badge.questionmark",
              url: "https://huggingface.co/datasets/liumindmind/NekoQA-10K"),
        .init(name: "RWKV Runner",
              blurb: "导出的 .pth 初始 state 挂载目标，与 RWKV 生态直连",
              license: "MIT",
              icon: "arrow.down.doc",
              url: "https://github.com/josStorer/RWKV-Runner"),
        .init(name: "Flash Linear Attention",
              blurb: "线性注意力上游库，模型转换校验基准",
              license: "MIT",
              icon: "bolt",
              url: "https://github.com/fla-org/flash-linear-attention"),
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
                .opacity(showEgg ? 0 : 1)
                .animation(.easeInOut(duration: 0.3), value: showEgg)

            HStack(spacing: 6) {
                Image("RWKVLogo")
                    .renderingMode(.template)
                    .resizable()
                    .scaledToFit()
                    .frame(width: 15, height: 15)
                    .onTapGesture { handleRavenTap() }
                    .background {
                        RavenPopoverPresenter(
                            isPresented: $showEgg,
                            text: eggMessage
                        )
                    }

                Text("RWKV-7 State Tuning")
            }
            .font(.callout)
            .foregroundStyle(.secondary)
            .animation(.easeInOut(duration: 0.2), value: showEgg)

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

    // MARK: - 渡鸦彩蛋

    /// 渡鸦彩蛋:前 7 下(RWKV-7 的 7)是解锁门槛;解锁后每点一下切换一条文案,
    /// 直到停手 3.5 秒气泡自动收起——收起即重新锁定,要再玩得重新点满 7 下。
    /// 首次解锁固定弹压轴句(Make RWKV Great Again),之后随机抽取且避免连抽同一条。
    private func handleRavenTap() {
        // 已解锁:每点一下换一条文案,气泡保持开,只重排收起计时。
        if eggUnlocked {
            pickNextEgg()
            scheduleDismiss()
            return
        }

        let now = Date()
        // 上一击已超时 → 从 1 重新计数。
        if now.timeIntervalSince(lastTapAt) > 1.5 {
            ravenTaps = 0
        }
        lastTapAt = now
        ravenTaps += 1

        guard ravenTaps == 7 else { return }
        ravenTaps = 0
        eggUnlocked = true

        // 首次解锁:固定压轴句。
        eggMessage = Self.eggMessages[0]
        lastEggIndex = 0
        showEgg = true
        scheduleDismiss()
    }

    /// 从压轴句之外随机抽一条,且避免与上次相同。
    private func pickNextEgg() {
        let pool = Array(Self.eggMessages.indices.dropFirst())
        guard !pool.isEmpty else { return }
        var idx = pool.randomElement()!
        if idx == lastEggIndex, pool.count > 1 {
            idx = pool.filter { $0 != lastEggIndex }.randomElement()!
        }
        lastEggIndex = idx
        eggMessage = Self.eggMessages[idx]
    }

    /// 排/重排 3.5 秒后自动收起气泡;连点换条时取消旧计时防止提前关闭。
    /// 收起时清空解锁态,下一次需重新点满 7 下。
    private func scheduleDismiss() {
        dismissWork?.cancel()
        let work = DispatchWorkItem {
            eggUnlocked = false
            showEgg = false
        }
        dismissWork = work
        DispatchQueue.main.asyncAfter(deadline: .now() + 3.5, execute: work)
    }

    // MARK: - 辅助

    private func open(_ urlString: String) {
        if let url = URL(string: urlString) {
            NSWorkspace.shared.open(url)
        }
    }
}

// MARK: - 渡鸦原生气泡

/// 用 `NSPopover` 承载彩蛋文案:气泡、箭头、材质和阴影均由 macOS 绘制。
/// `.applicationDefined` 让连点渡鸦时气泡保持展开,只由外层 3.5 秒计时器收起。
private struct RavenPopoverPresenter: NSViewRepresentable {
    @Binding var isPresented: Bool
    let text: String

    func makeCoordinator() -> Coordinator {
        Coordinator(isPresented: $isPresented)
    }

    func makeNSView(context: Context) -> NSView {
        NSView(frame: .zero)
    }

    func updateNSView(_ nsView: NSView, context: Context) {
        context.coordinator.isPresented = $isPresented
        context.coordinator.update(text: text)

        if isPresented {
            context.coordinator.present(from: nsView)
        } else {
            context.coordinator.dismiss()
        }
    }

    static func dismantleNSView(_ nsView: NSView, coordinator: Coordinator) {
        coordinator.dismiss()
    }

    @MainActor
    final class Coordinator: NSObject, NSPopoverDelegate {
        var isPresented: Binding<Bool>

        private let popover: NSPopover
        private let hostingController: NSHostingController<RavenPopoverContent>
        private var currentText = ""

        init(isPresented: Binding<Bool>) {
            self.isPresented = isPresented
            hostingController = NSHostingController(
                rootView: RavenPopoverContent(text: "")
            )
            popover = NSPopover()
            super.init()

            popover.animates = true
            popover.behavior = .applicationDefined
            popover.contentViewController = hostingController
            popover.delegate = self
        }

        func update(text: String) {
            guard text != currentText else { return }
            currentText = text
            hostingController.rootView = RavenPopoverContent(text: text)
            hostingController.view.layoutSubtreeIfNeeded()
            popover.contentSize = hostingController.view.fittingSize
        }

        func present(from anchor: NSView) {
            guard !popover.isShown, anchor.window != nil else { return }
            popover.show(
                relativeTo: anchor.bounds,
                of: anchor,
                preferredEdge: .maxY
            )
        }

        func dismiss() {
            guard popover.isShown else { return }
            popover.performClose(nil)
        }

        func popoverDidClose(_ notification: Notification) {
            if isPresented.wrappedValue {
                isPresented.wrappedValue = false
            }
        }
    }
}

private struct RavenPopoverContent: View {
    let text: String

    var body: some View {
        Text(text)
            .font(.callout.weight(.medium))
            .padding(.horizontal, 12)
            .padding(.vertical, 7)
            .fixedSize()
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

                Text(L10n.string(credit.blurb))
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
