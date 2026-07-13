import SwiftUI

/// 将同一区域内的玻璃面统一交给系统合成；旧系统保持原视图不变。
struct PreenGlassEffectGroup<Content: View>: View {
    let spacing: CGFloat
    private let content: Content

    init(spacing: CGFloat = 8, @ViewBuilder content: () -> Content) {
        self.spacing = spacing
        self.content = content()
    }

    @ViewBuilder
    var body: some View {
        if #available(macOS 26.0, *) {
            GlassEffectContainer(spacing: spacing) {
                content
            }
        } else {
            content
        }
    }
}

extension View {
    /// 只在 macOS 26+ 添加真正的 Liquid Glass，避免改变旧系统现有外观。
    @ViewBuilder
    func preenGlassSurface(cornerRadius: CGFloat = 14, interactive: Bool = false) -> some View {
        if #available(macOS 26.0, *) {
            glassEffect(
                interactive ? .regular.interactive() : .regular,
                in: RoundedRectangle(cornerRadius: cornerRadius, style: .continuous)
            )
        } else {
            self
        }
    }

    /// 主操作在新系统使用玻璃按钮，旧系统继续使用原有 bordered 样式。
    @ViewBuilder
    func preenGlassButton(prominent: Bool = false) -> some View {
        if #available(macOS 26.0, *) {
            if prominent {
                buttonStyle(.glassProminent)
            } else {
                buttonStyle(.glass)
            }
        } else if prominent {
            buttonStyle(.borderedProminent)
        } else {
            buttonStyle(.bordered)
        }
    }

    /// 快捷卡片在新系统由自定义玻璃面承载，旧系统保留 bordered 卡片。
    @ViewBuilder
    func preenQuickActionButtonStyle() -> some View {
        if #available(macOS 26.0, *) {
            buttonStyle(.plain)
        } else {
            buttonStyle(.bordered)
        }
    }
}
