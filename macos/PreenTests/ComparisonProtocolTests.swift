import XCTest
@testable import Preen

final class ComparisonProtocolTests: XCTestCase {
    func testSideTaggedStreamingEventsDecodeIndependently() throws {
        let chunk = try JSONDecoder().decode(
            ServeEvent.self,
            from: Data(#"{"type":"text_chunk","id":"ab1","side":"baseline","delta":"base","phase":"answer"}"#.utf8)
        )
        guard case .textChunk(let id, _, let side, let delta, let phase) = chunk else {
            return XCTFail("Expected text_chunk")
        }
        XCTAssertEqual(id, "ab1")
        XCTAssertEqual(side, .baseline)
        XCTAssertEqual(delta, "base")
        XCTAssertEqual(phase, .answer)

        let error = try JSONDecoder().decode(
            ServeEvent.self,
            from: Data(#"{"type":"side_error","id":"ab1","side":"with_state","code":"internal","message":"bad state"}"#.utf8)
        )
        guard case .sideError(_, let errorSide, let code, let message) = error else {
            return XCTFail("Expected side_error")
        }
        XCTAssertEqual(errorSide, .withState)
        XCTAssertEqual(code, .internal)
        XCTAssertEqual(message, "bad state")
        XCTAssertFalse(error.isTerminal, "side_error must not consume the preview continuation")

        let terminal = try JSONDecoder().decode(
            ServeEvent.self,
            from: Data(#"{"type":"ok","id":"ab1"}"#.utf8)
        )
        XCTAssertTrue(terminal.isTerminal)
    }

    func testPreviewRequestCarriesStateFormatAndSevenGenerationFields() throws {
        let config = ChatSessionConfig(
            template: .instruction,
            reasoning: false,
            think: .off,
            genConfig: .defaultConfig
        )
        let request = ServeRequest.preview(
            id: "ab2",
            prompt: "hello",
            template: config.template.rawValue,
            reasoning: config.reasoning,
            think: config.think.rawValue,
            statePath: "/tmp/state.npz",
            ab: true,
            genConfig: config.genConfig.toDTO()
        )
        let object = try XCTUnwrap(
            JSONSerialization.jsonObject(with: Data(request.encodeToLine().utf8)) as? [String: Any]
        )
        XCTAssertEqual(object["template"] as? String, "instruction")
        XCTAssertEqual(object["state_path"] as? String, "/tmp/state.npz")
        XCTAssertEqual(object["ab"] as? Bool, true)
        let generation = try XCTUnwrap(object["gen_config"] as? [String: Any])
        XCTAssertEqual(generation.count, 7)
        XCTAssertEqual(generation["presence_penalty"] as? Double, 0.4)
        XCTAssertEqual(generation["frequency_penalty"] as? Double, 0.4)
        XCTAssertEqual(generation["penalty_decay"] as? Double, 0.996)
    }
}
