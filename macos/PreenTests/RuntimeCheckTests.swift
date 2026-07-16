import XCTest
@testable import Preen

final class RuntimeCheckTests: XCTestCase {
    func testSuccessfulDoctorFixture() throws {
        let result = RuntimeCheckResult.decode(
            output: Data(successJSON.utf8), stderr: Data(),
            exit: ProcessExitInfo(status: 0, reason: .exit)
        )
        XCTAssertEqual(result.report?.python, "3.11.15")
        XCTAssertEqual(result.report?.chipName, "Apple M5")
        XCTAssertEqual(result.report?.hardwareModel, "Mac17,3")
        XCTAssertEqual(result.report?.memorySizeLabel, "16 GB")
        XCTAssertEqual(result.report?.workingSetLabel, "11.84 GB")
        XCTAssertEqual(result.report?.isUsable, true)
        XCTAssertNil(result.errorMessage)
    }

    func testDiagnosticMarkdownIsIssueReadyAndExcludesArbitraryMessages() throws {
        let result = RuntimeCheckResult.decode(
            output: Data(successJSON.utf8), stderr: Data(),
            exit: ProcessExitInfo(status: 0, reason: .exit)
        )
        let runtime = RuntimeStatus(
            phase: .ready,
            report: try XCTUnwrap(result.report),
            message: "来自 /Users/alice/private 的运行时",
            checkedAt: nil
        )
        let markdown = BackendDiagnostics.markdown(
            runtime: runtime,
            inference: WorkerStatus(
                phase: .failed, pid: 123, message: "模型位于 /Users/alice/secret"
            ),
            training: WorkerStatus(
                phase: .idle, pid: nil, message: "数据位于 /Users/alice/dataset"
            ),
            appVersion: "1.0",
            appBuild: "1",
            systemVersionFallback: "macOS",
            generatedAt: Date(timeIntervalSince1970: 0)
        )

        XCTAssertTrue(markdown.contains(L10n.format("- 芯片: %@", "Apple M5")))
        XCTAssertTrue(
            markdown.contains(L10n.format("- 系统: %@", "macOS 26.5.2 (25F84)"))
        )
        XCTAssertFalse(markdown.contains("- macOS: macOS"))
        XCTAssertTrue(markdown.contains(L10n.format("- 统一内存: %@", "16 GB")))
        XCTAssertTrue(
            markdown.contains(L10n.format("- MLX 建议工作集上限: %@", "11.84 GB"))
        )
        XCTAssertTrue(
            markdown.contains(L10n.format("- 推理服务: %@", L10n.string("失败")))
        )
        XCTAssertFalse(markdown.contains("/Users/alice"))
        XCTAssertFalse(markdown.contains("123"))
    }

    func testMissingPythonFixture() {
        let result = RuntimeCheckResult.decode(
            output: Data(), stderr: Data("python executable not found".utf8), exit: nil
        )
        XCTAssertNil(result.report)
        XCTAssertEqual(result.errorMessage, "python executable not found")
    }

    func testDoctorDecimalGBIsConvertedForAppDisplay() {
        let legacyJSON = successJSON
            .replacingOccurrences(
                of: ",\"os_version\":\"26.5.2\",\"os_build\":\"25F84\",\"chip_name\":\"Apple M5\",\"hardware_model\":\"Mac17,3\"",
                with: ""
            )
            .replacingOccurrences(of: ",\"memory_size_gib\":16.0", with: "")
            .replacingOccurrences(of: ",\"working_set_gib\":11.84", with: "")
        let result = RuntimeCheckResult.decode(
            output: Data(legacyJSON.utf8), stderr: Data(), exit: nil
        )

        XCTAssertEqual(result.report?.memorySizeGB, 17.18)
        XCTAssertEqual(result.report?.memorySizeLabel, "16 GB")
        XCTAssertEqual(result.report?.workingSetLabel, "11.84 GB")
        XCTAssertEqual(result.report?.operatingSystemLabel, "macOS")
    }

    func testMissingMLXFixture() {
        let json = successJSON.replacingOccurrences(
            of: "\"mlx\":{\"ok\":true,\"version\":\"unknown\",\"error\":null}",
            with: "\"mlx\":{\"ok\":false,\"version\":null,\"error\":\"No module named mlx\"}"
        )
        let result = RuntimeCheckResult.decode(output: Data(json.utf8), stderr: Data(), exit: nil)
        XCTAssertEqual(result.report?.isUsable, false)
        XCTAssertEqual(result.errorMessage, "No module named mlx")
    }

    private var successJSON: String {
        """
        {"python":"3.11.15","platform":"macOS","machine":"arm64","apple_silicon":true,"os_version":"26.5.2","os_build":"25F84","chip_name":"Apple M5","hardware_model":"Mac17,3","numpy":{"ok":true,"version":"2.4","error":null},"ml_dtypes":{"ok":true,"version":"0.5","error":null},"mlx":{"ok":true,"version":"unknown","error":null},"mlx_lm":{"ok":true,"version":"0.31","error":null},"metal_available":true,"metal_error":null,"memory_size_gb":17.18,"memory_size_gib":16.0,"working_set_gb":12.71,"working_set_gib":11.84}
        """
    }
}
