//
//  PythonResolver.swift
//  Preen
//
//  Sidecar Python 解释器路径解析 + 子进程环境注入。
//
//  解析顺序(开发/发布共用同一套代码,环境变量切换):
//   1. PREEN_SIDECAR_PYTHON 环境变量(开发态 → 本地 uv venv 的 .venv/bin/python3)
//   2. Bundle.main 内 python-build-standalone(发布态)
//
//  仓库根(开发态 PYTHONPATH=src 需要)同理:
//   1. PREEN_REPO_ROOT 环境变量
//   2. 从 PREEN_SIDECAR_PYTHON 反推(.venv/bin/python3 → 父父父父 = repo root)
//   3. 发布态为 nil(靠 site-packages,不靠 PYTHONPATH)
//

import Foundation

/// 解析 sidecar Python 解释器与运行环境。
///
/// 所有方法都不做 IO(不检查文件存在性),只做路径推导 ——
/// 调用方 spawn 时若路径不对,Process 会自然失败,stderr 会带原因。
enum PythonResolver {

    /// Sidecar 解释器可执行文件 URL。
    /// 开发态:`PREEN_SIDECAR_PYTHON`(指向 .venv/bin/python3)。
    /// 发布态:`Bundle.main/python/bin/python3`。
    static var executable: URL {
        if let env = ProcessInfo.processInfo.environment["PREEN_SIDECAR_PYTHON"], !env.isEmpty {
            return URL(fileURLWithPath: env)
        }
        return Bundle.main.resourceURL?
            .appendingPathComponent("python", isDirectory: true)
            .appendingPathComponent("bin", isDirectory: true)
            .appendingPathComponent("python3", isDirectory: false)
            ?? URL(fileURLWithPath: "/usr/bin/python3")
    }

    /// 仓库根目录(开发态,用于 `PYTHONPATH=src`)。
    /// 从 `PREEN_REPO_ROOT` 或 `.venv/bin/python3` 反推;发布态为 nil。
    static var repoRoot: URL? {
        let env = ProcessInfo.processInfo.environment
        if let root = env["PREEN_REPO_ROOT"], !root.isEmpty {
            return URL(fileURLWithPath: root)
        }
        // .venv/bin/python3 → ../.. = repo root(.venv 在 repo 根下)
        if env["PREEN_SIDECAR_PYTHON"] != nil {
            return executable
                .deletingLastPathComponent()  // bin
                .deletingLastPathComponent()  // .venv
        }
        return nil
    }

    /// 应用数据根目录(放 models / states / datasets / hf-cache)。
    /// `~/Library/Application Support/Preen/`。
    static var dataRoot: URL {
        let fm = FileManager.default
        let support = fm.urls(for: .applicationSupportDirectory, in: .userDomainMask).first
            ?? URL(fileURLWithPath: NSHomeDirectory()).appendingPathComponent("Library/Application Support")
        let root = support.appendingPathComponent("Preen", isDirectory: true)
        // 确保目录存在(幂等)。
        try? fm.createDirectory(at: root, withIntermediateDirectories: true)
        return root
    }

    /// HF 缓存目录(`HF_HOME` 指这里,禁止污染 ~/.cache)。
    static var hfCache: URL {
        dataRoot.appendingPathComponent("hf-cache", isDirectory: true)
    }

    /// spawn 子进程时注入的环境变量。
    /// 在父进程环境基础上覆盖:
    /// - `PYTHONPATH=<repo>/src`(仅开发态有 repoRoot)
    /// - `HF_HOME=<data>/hf-cache`
    /// - `LC_ALL` / `LANG` = en_US.UTF-8(stdout JSON 用 ASCII,但 stderr 人类日志可能含中文)
    static var childEnvironment: [String: String] {
        var env = ProcessInfo.processInfo.environment
        if let root = repoRoot {
            env["PYTHONPATH"] = root.appendingPathComponent("src").path
        } else {
            env.removeValue(forKey: "PYTHONPATH")
        }
        env["HF_HOME"] = hfCache.path
        env["LC_ALL"] = "en_US.UTF-8"
        env["LANG"] = "en_US.UTF-8"
        // 让 Python 输出不缓冲(Swift 按行读 stdout,缓冲会延迟事件)。
        env["PYTHONUNBUFFERED"] = "1"
        return env
    }
}
