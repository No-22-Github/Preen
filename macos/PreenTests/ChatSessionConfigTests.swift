import XCTest
@testable import Preen

final class ChatSessionConfigTests: XCTestCase {
    func testSafeDefaultMatchesServeContract() {
        let config = ChatSessionConfig.defaultConfig
        XCTAssertEqual(config.template, .qa)
        XCTAssertFalse(config.reasoning)
        XCTAssertEqual(config.think, .off)
        XCTAssertTrue(config.isValid)
    }

    func testNonQATemplateNormalizesReasoningDialect() {
        let config = ChatSessionConfig(
            template: .instruction,
            reasoning: true,
            think: .on,
            genConfig: .defaultConfig
        ).normalized()
        XCTAssertEqual(config.template, .instruction)
        XCTAssertFalse(config.reasoning)
        XCTAssertEqual(config.think, .off)
    }

    func testReasoningThinkModesRemainSelectableForQA() {
        for mode in ThinkMode.allCases {
            let config = ChatSessionConfig(
                template: .qa,
                reasoning: true,
                think: mode,
                genConfig: .defaultConfig
            )
            XCTAssertTrue(config.isValid, "QA reasoning should accept \(mode.rawValue)")
            XCTAssertEqual(config.normalized(), config)
        }
    }

    func testNewSessionRequestCarriesExplicitFormatAndAllSamplingFields() throws {
        let config = ChatSessionConfig(
            template: .qa,
            reasoning: true,
            think: .fast,
            genConfig: .defaultConfig
        )
        let request = ServeRequest.newSession(
            id: "session-format",
            template: config.template.rawValue,
            reasoning: config.reasoning,
            think: config.think.rawValue,
            genConfig: config.genConfig.toDTO()
        )
        let object = try XCTUnwrap(
            JSONSerialization.jsonObject(with: Data(request.encodeToLine().utf8)) as? [String: Any]
        )
        XCTAssertEqual(object["template"] as? String, "qa")
        XCTAssertEqual(object["reasoning"] as? Bool, true)
        XCTAssertEqual(object["think"] as? String, "fast")
        let generation = try XCTUnwrap(object["gen_config"] as? [String: Any])
        XCTAssertEqual(Set(generation.keys), Set([
            "max_tokens", "temperature", "top_p", "seed",
            "presence_penalty", "frequency_penalty", "penalty_decay"
        ]))
    }
}
