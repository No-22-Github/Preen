import Foundation
import UserNotifications

final class TrainingNotificationController {
    func send(title: String, body: String) {
        let center = UNUserNotificationCenter.current()
        center.requestAuthorization(options: [.alert, .sound]) { granted, _ in
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
}
