import Foundation

/// App localization entry point for strings that are not created by SwiftUI.
///
/// SwiftUI localizes literal labels automatically. Status models, file panels,
/// notifications, and formatted messages produce plain `String` values, so they
/// must opt in explicitly. Language selection and fallback remain entirely under
/// `Bundle`/macOS control; the app never inspects or overrides the current locale.
enum L10n {
    static func string(_ key: String) -> String {
        Bundle.main.localizedString(forKey: key, value: key, table: nil)
    }

    static func format(_ key: String, _ arguments: CVarArg...) -> String {
        String(format: string(key), locale: Locale.current, arguments: arguments)
    }

    /// Python diagnostics are intentionally English. Keep their detail in an English UI, while
    /// a Chinese UI uses a natural localized summary when no exact translation exists.
    static func backendMessage(_ message: String, fallback fallbackKey: String) -> String {
        let localized = string(message)
        if localized != message || Bundle.main.preferredLocalizations.first?.hasPrefix("en") == true {
            return localized
        }
        return string(fallbackKey)
    }
}
