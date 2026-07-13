import XCTest
@testable import Preen

final class RuntimeCheckTests: XCTestCase {
    func testSuccessfulDoctorFixture() throws {
        let result = RuntimeCheckResult.decode(
            output: Data(successJSON.utf8), stderr: Data(),
            exit: ProcessExitInfo(status: 0, reason: .exit)
        )
        XCTAssertEqual(result.report?.python, "3.11.15")
        XCTAssertEqual(result.report?.isUsable, true)
        XCTAssertNil(result.errorMessage)
    }

    func testMissingPythonFixture() {
        let result = RuntimeCheckResult.decode(
            output: Data(), stderr: Data("python executable not found".utf8), exit: nil
        )
        XCTAssertNil(result.report)
        XCTAssertEqual(result.errorMessage, "python executable not found")
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
        {"python":"3.11.15","platform":"macOS","machine":"arm64","apple_silicon":true,"numpy":{"ok":true,"version":"2.4","error":null},"ml_dtypes":{"ok":true,"version":"0.5","error":null},"mlx":{"ok":true,"version":"unknown","error":null},"mlx_lm":{"ok":true,"version":"0.31","error":null},"metal_available":true,"metal_error":null,"memory_size_gb":17.18,"working_set_gb":12.71}
        """
    }
}
