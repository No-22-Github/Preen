import Foundation
import UserNotifications

final class TrainingNotificationController: NSObject, UNUserNotificationCenterDelegate {
    static let authorizationOptions: UNAuthorizationOptions = [.alert, .sound, .badge]

    override init() {
        super.init()
        UNUserNotificationCenter.current().delegate = self
    }

    /// 在训练开始时注册通知能力，让系统开放 Dock「标记」开关。
    func prepare() {
        UNUserNotificationCenter.current().requestAuthorization(
            options: Self.authorizationOptions
        ) { _, _ in }
    }

    func send(title: String, body: String) {
        let center = UNUserNotificationCenter.current()
        center.requestAuthorization(options: Self.authorizationOptions) { granted, _ in
            guard granted else { return }
            let content = UNMutableNotificationContent()
            content.title = title
            content.body = body
            content.sound = .default
            center.add(UNNotificationRequest(
                identifier: "preen-training-\(UUID().uuidString)", content: content, trigger: nil
            ))
        }
    }

    func userNotificationCenter(
        _ center: UNUserNotificationCenter,
        willPresent notification: UNNotification,
        withCompletionHandler completionHandler: @escaping (UNNotificationPresentationOptions) -> Void
    ) {
        completionHandler([.banner, .sound])
    }
}
