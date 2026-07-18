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
        XCTAssertTrue(intent.title(isGenerating: true).contains("停止"))
        let message = intent.consequence(currentModelPath: "/models/old", isGenerating: true)
        XCTAssertTrue(message.contains("rwkv7-g1d"))
        XCTAssertTrue(message.contains("当前生成"))
        XCTAssertEqual(intent.destructiveButtonTitle, L10n.string("切换模型"))
    }
}
