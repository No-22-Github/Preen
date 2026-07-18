# Backlog：签名、公证与无终端安装

> 优先级：未排期  
> 状态：已延期  
> 所属版本：后续待定  
> 裁决：2026-07-18 明确不纳入 v1.1，不作为当前发布条件。

## 一、问题

当前公开包未使用 Developer ID 签名和 Apple 公证。用户首次打开需要在终端执行 `xattr`，这对非开发者是显著的信任与转化障碍，也削弱了原生 macOS 产品的完整感。

## 二、目标

- 用户下载、解压后可直接双击启动。
- 发布物可由 Gatekeeper 正常验证。
- 保持 macOS 14 / macOS 26 两个构建目标的现有功能一致。
- 构建与发布过程可重复、可审计。

## 三、前置条件

- Apple Developer Program 账号。
- Developer ID Application 证书。
- 公证凭据以 CI secret 保存。
- 明确团队 ID、Bundle ID 和 entitlements。

若前置条件未满足，本需求不得伪装完成。

## 四、发布流程

1. 构建自包含 App。
2. 按从内到外顺序，对嵌入的 Python 可执行文件、动态库、framework 与主 App 分别签名；不依赖单独一次 `--deep` 自动修补嵌套代码。
3. 使用 Developer ID Application 、Hardened Runtime 与 secure timestamp 签名，只添加运行必需的 entitlements。
4. 用 `ditto` 生成仅供公证上传的 ZIP，使用 `notarytool` 提交并等待 Accepted，同时保存与检查 notarization log。
5. ZIP 不能直接 staple；对 ZIP 内对应的 `Preen.app` 执行 `stapler staple` 和 `stapler validate`，再将已附票的 App 重新打包为最终发行 ZIP。若改用 DMG，则按 DMG 容器的签名、公证与 staple 流程执行。
6. 对最终发行 ZIP 重新计算 SHA-256，在隔离干净机器或新用户环境从浏览器下载并验证首次启动。
7. 发布 manifest 记录版本、构建目标、最终 SHA、签名身份、公证 submission ID 和结果。

## 五、安装体验

- README 主流程改为“下载 → 解压 → 打开”。
- `xattr` 只保留在历史版本或排障附录，不作为主安装步骤。
- 如果系统版本不满足要求，App 或 Finder 给出明确兼容性信息。
- 不自行设计安装器；ZIP 仍可作为首选分发格式。

## 六、验收标准

- [ ] `codesign --verify --deep --strict` 通过。
- [ ] 主 App、内嵌 Python 与关键 `.dylib/.so/framework` 可分别验证签名身份、Hardened Runtime 与 timestamp，不只依赖 `--deep` 的整体返回值。
- [ ] `spctl --assess --type execute` 通过。
- [ ] `stapler validate` 通过。
- [ ] 最终分发 ZIP 中的 App 是已 staple 的版本，不把仅供上传的未附票 ZIP 误当成发行物。
- [ ] 从浏览器下载后保留 quarantine 的 App 可直接双击启动。
- [ ] 内嵌 Python 能正常运行 runtime check、训练和推理 smoke test。
- [ ] macOS 14 与 macOS 26 两个发布包分别在目标系统验证。
- [ ] CI 不在日志中泄露公证凭据。

## 七、不做

- 不在本需求上架 Mac App Store。
- 不引入自动更新框架。
- 不承诺 Intel 支持。
