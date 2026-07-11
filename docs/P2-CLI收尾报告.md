# P2 CLI 收尾报告 — 命令逐项修复与解耦

> 日期：2026-07-12 ｜ 阶段：P2 已完成（CLI + 独立推理引擎）
> 范围：按命令逐项审计/修复，目标——cli.py 是薄壳，业务编排在 service / inference / inspection。
> 验收：删掉 cli.py 后，SwiftUI sidecar 仅依赖 service/inference/inspection/events 即可实现全部同等功能（终端 REPL 除外）。

---

## 1. doctor — 审计，无需改动

`inspection.doctor_report()` 返回完整 dict；`--json` 走 `json.dumps(report)` 是文本展示的超集
（含 python/platform/machine/apple_silicon/numpy/torch/mlx/mlx_lm/metal_available/memory/working_set）。
帮助文本准确。报告型命令恒 exit 0，语义保持。

## 2. data-info — 审计，无需改动

`inspection.inspect_data()` 薄壳；`--json` 输出 `to_dict()`（asdict 全字段）。
可预期错误（模型/数据缺失、ctx-len 非法、tokenizer OSError/ValueError/TypeError）经 `_bad_input` → exit 2 + stderr。

## 3. state-info — 审计，无需改动

`inspection.inspect_state()` 薄壳；`--json` 输出 `to_dict()` 全字段。
错误（OSError/ValueError/TypeError/KeyError）经 `_bad_input` → exit 2 + stderr。

## 4. train — 审计 + 补测试 + TODO 注记

**审计结论**：事件流 JSON lines → stdout、`#` 状态提示 → stderr（`err=True`），无混淆。
退出路径 2（输入错误）/1（运行失败）/130（中断）三类齐全。

**改动**：
- 补 130（KeyboardInterrupt）与 1（run_training 抛异常）两条退出路径的快测
  （`test_train_interrupt_exits_130` / `test_train_runtime_failure_exits_1`，monkeypatch run_training，不加载真实模型）。
  原有 exit 2 路径已有覆盖。
- 在 `if template != "nekoqa"` 校验前加 TODO 注记：g1g 训练模板是否开放是未做的产品决策
  （reasoning 模板的 state tuning 风格/格式注入能力未验证）。

**不改**：Trainer 主体、loss 计算、超参默认值、CLI/service 双层校验（有意设计，双层防御）。

## 5. eval — 修 bug + 编排下沉（本次核心）

**Bug**：`cli.py` 中 `prompt = render_prompt(q, "nekoqa")` 硬编码模板，
导致 `--template g1g` 时 stop sequences 用 g1g、prompt 渲染用 nekoqa，不同源。
**修复**：改为 `render_prompt(question, request.template)`，渲染与 stops 同源。

**下沉**（service.py 新增）：
- `EvaluationRequest`（frozen dataclass）：engine / state / template / config / data / limit。
- `EvaluationResult` / `EvaluationItem`：结构化结果，供文本/JSON 双输出派生。
- `validate_evaluation_request`：独立前置校验（不依赖 CLI）。
- `run_evaluation`：QA 加载（load_qa_pairs + 错误转译）、缺省示例 prompt 列表（`DEFAULT_EVAL_QUESTIONS`，
  从 cli.py 迁入——属用例不属 CLI）、逐条生成循环 + limit 截断 + 结构化结果组装。不 print。
- **engine 由调用方注入**（模型加载留在 CLI），使 service 层可用 FakeEngine 单测。

**CLI 层保留**：参数校验 → 加载模型 + 构造 engine → 构造 Request → 调 service → 文本或 JSON 输出。

**测试**（新增 `tests/test_eval.py`，8 例）：
- ① 回归：nekoqa / g1g 两条 template 各自证明 `render_prompt` 输出与 `config.stop_sequences`
  同源（FakeEngine 记录调用，不加载真实模型）——这是历史 bug 的防线。
- ② service 层单测：数据加载失败（ValueError）、limit 截断（5 条数据 limit=2 只生成 2 条）、
  无 data 用内置示例、结果结构、validate 拒绝坏 template / 缺 state。

## 6. export — 审计，无需改动

薄壳调 `export.py`；错误路径齐全：state 缺失/非 .npz → exit 2 + stderr，
round-trip 失败 → exit 1，成功 → stdout。帮助文本准确。

## 7. chat — state 校验下沉，REPL 保留

**下沉**：cli.py 内联的 `_load_checked`（层数/shape/rwkv7_compatible 校验）
移到 `inspection.validate_state_for_model(state_path, model) -> dict`。
CLI 初始加载与 ChatSession 运行中 `/state` 切换共用同一函数。
**行为与错误信息逐字不变**（三重校验 + 相同 ValueError 文案）。

**明确保留在 CLI**：`input()` 的 REPL 循环、流式 echo 回调（`_on_text`）——
终端特有交互，属 CLI 层合法职责，未下沉、未新建抽象。

**测试**（`tests/test_inspection.py` 新增 4 例，tmp npz + FakeModel，不加载真实模型）：
层数匹配通过 / 层数不匹配拒绝 / shape 不匹配拒绝 / 非 rwkv7 格式拒绝。

## 8. preview — 消重复，其余保持

**改动**：`[stop=..., tokens=..., t/s]` 摘要行原在 cli.py（preview 的 stream 与 non-stream 两处）
和 chat.py `_summary` 各写一份。新增 `GenerationResult.summary_line() -> str` 作为单一来源，
三处改为调用。**输出字符串逐字符不变**（test_chat 的精确断言 + 新增格式锁定测试均通过）。

**核对**：`--ab` / `--stream` / `--json` 互斥校验齐全且已有测试覆盖
（`test_preview_ab_requires_state` / `test_preview_stream_rejects_json`；`--stream + --ab` 同一分支）。
A/B 的短摘要 `[stop=..., tokens=...]` 是另一种展示（无 tps），未动。

---

## 收尾扫描发现

- **preview `--template` 帮助文本漏列 g1g**：代码接受 g1g 但帮助只写 `raw | nekoqa`。
  已修正为 `raw(原样,默认) | nekoqa | g1g(RWKV7-G1 原生)`，docstring 同步。**小且无争议。**
- **exit code 约定全命令一致**：0=成功 / 2=输入错误（`_bad_input`）/ 1=运行失败 / 130=中断。
- **stdout/stderr 分离干净**：`#` 状态/进度/错误 → stderr（`err=True`）；JSON/结果文本 → stdout。
- **`--template` 支持范围**（有意设计，未改，仅核对描述准确）：
  train=nekoqa；eval=nekoqa/g1g；chat=raw/nekoqa/g1g（默认 g1g）；preview=raw/nekoqa/g1g（默认 raw）。
- **发现但未动手（超出清单范围）**：`data.py` 末尾有 7 处 `print()` 调用（`--- 样本 {i} ---` 格式的
  调试辅助函数）。不在任何 CLI 命令路径上，sidecar 不会触发，属开发调试代码。记录不动。

## 文档同步（仅事实性陈述）

- `AGENTS.md`：「当前阶段 P1」→「P2 已完成」；「P1 正式代码在 p1-statetuner-cli 分支」
  →「正式代码全部在 main 分支；唯一存活实验分支是 exp/precision」。与 README 一致。

---

## 验收对照

| 标准 | 结果 |
|---|---|
| `.venv/bin/python -m pytest -q` 全绿 | ✅ 73 passed, 3 skipped |
| `--slow` 确认（本机有模型） | ✅ 76 passed（含 golden 逐字零回退 + 训练行为断言） |
| eval 模板回归测试存在且通过 | ✅ `test_eval_nekoqa/g1g_prompt_and_stops_share_template` |
| cli.py 无 sidecar 需复用的业务编排（REPL 除外） | ✅ eval 编排在 service；chat state 校验在 inspection；preview 摘要在 inference |
| `GenerationResult.summary_line()` 摘要行唯一来源 | ✅ cli.py + chat.py 共用，输出逐字符不变 |
| golden 快照零回退 | ✅ `--slow` 通过 |

## 变更文件

```
AGENTS.md                      分支/阶段事实修正
src/statetuner/cli.py          eval 下沉 + chat 校验引用 + preview 摘要引用 + 帮助文本修正
src/statetuner/service.py      新增 EvaluationRequest/run_evaluation/validate（+100 行）
src/statetuner/inference.py    新增 GenerationResult.summary_line()
src/statetuner/inspection.py   新增 validate_state_for_model()
src/statetuner/chat.py         _summary 委派 summary_line()
tests/test_cli.py              +2 退出路径测试（130/1）
tests/test_eval.py             新增（8 例：模板同源回归 + service 单测）
tests/test_inference_engine.py +summary_line 格式锁定测试
tests/test_inspection.py       +4 validate_state_for_model 测试
```
