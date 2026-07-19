import XCTest
@testable import Preen

final class SessionReplacementPolicyTests: XCTestCase {
    func testEmptySessionDoesNotInterruptOneClickEntry() {
        XCTAssertFalse(SessionReplacementPolicy.requiresConfirmation(
            messageCount: 0,
            comparisonHasContent: false,
            isGenerating: false
        ))
    }

    func testEveryUserOwnedSessionStateRequiresConfirmation() {
        XCTAssertTrue(SessionReplacementPolicy.requiresConfirmation(
            messageCount: 2,
            comparisonHasContent: false,
            isGenerating: false
        ))
        XCTAssertTrue(SessionReplacementPolicy.requiresConfirmation(
            messageCount: 0,
            comparisonHasContent: true,
            isGenerating: false
        ))
        XCTAssertTrue(SessionReplacementPolicy.requiresConfirmation(
            messageCount: 0,
            comparisonHasContent: false,
            isGenerating: true
        ))
    }

    func testGeneratingPresentationNamesAbortAndConcreteTarget() {
        let intent = SessionReplacementIntent.selectModel("/models/rwkv7-g1d")
        // 期望值从同一本地化表派生,locale 中立(en runner 与 zh-Hans 开发机下都稳定)。
        XCTAssertTrue(intent.title(isGenerating: true).contains(L10n.string("并停止当前生成？")))
        let message = intent.consequence(currentModelPath: "/models/old", isGenerating: true)
        XCTAssertTrue(message.contains("rwkv7-g1d"))
        XCTAssertTrue(message.contains(L10n.string("当前生成也会停止。")))
        XCTAssertEqual(intent.destructiveButtonTitle, L10n.string("切换模型"))
    }
}
