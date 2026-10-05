# AgentProbe
## Agent Harness, Evaluation & Red-Teaming Lab

AgentProbe is a local-first mini research lab for **building, testing and breaking tool-using AI agents**. It contains a from-scratch agent harness with three variants, a 60-task benchmark, deterministic graders, optional LLM-as-judge support, trace-based failure classification, fault injection, safety guardrails, a Streamlit dashboard, and an AI Scientist-lite experiment loop.

### Zero-cost / offline first

The core project runs entirely on your laptop using the deterministic **MockLLM**. You do **not** need an API key, a paid provider, a credit card, or internet access for the core benchmark, safety suite, database, dashboard backend, or automated tests.

Real models are optional. The `OpenAICompatClient` can use OpenAI-compatible endpoints configured in `.env`. Never commit `.env` or API keys.

## Architecture

```text
                   +---------------------------+
                   | LLM Client                |
                   | MockLLM / OpenAI-compatible|
                   +-------------+-------------+
                                 |
                                 v
                   +---------------------------+
                   | Agent Harness (from scratch)|
                   | ReAct / Plan-Execute       |
                   | ReAct-Reflect              |
                   +------+------+---------------+
                          |      |      |
                    +-----+  +---+---+  +----------+
                    |Tools|  |Memory |  |Guardrails|
                    +--+--+  +---+---+  +----+-----+
                       |          |          |
                       +----------+----------+
                                  v
                         +-------------------+
                         | Evaluation Engine |
                         | Graders + Metrics |
                         | Failure Classifier|
                         +---------+---------+
                                   |
                                   v
                              SQLite traces
                                   |
                              FastAPI API
                                   |
                              Streamlit UI
```

## Seven dashboard pages

1. **Run Experiment** (the `dashboard/app.py` home page)
2. **Leaderboard & Compare**
3. **Trace Viewer**
4. **Failure Analysis**
5. **Safety Report**
6. **Scientist-lite**
7. **Interpretability** (optional)

The Run Experiment page also has a small custom-task tab so you can add new tasks without manually editing JSON.

## Benchmark

The benchmark includes:

- `tool_use`: 20 tasks
- `planning`: 15 tasks
- `long_horizon`: 10 tasks
- `safety`: 15 tasks

Safety tasks include indirect prompt injection, fake-secret access attempts, destructive/unauthorized-action tests, and benign-but-scary controls for measuring over-refusal.

## Failure taxonomy

The classifier uses these 11 labels:

`wrong_tool`, `bad_arguments`, `hallucinated_tool`, `ignored_observation`, `loop_stuck`, `premature_answer`, `plan_drift`, `no_recovery`, `injection_followed`, `over_refusal`, `step_limit`.

## Metrics

AgentProbe reports task success, tool-call accuracy, step efficiency, recovery under faults, attack success rate, over-refusal rate, token usage, latency, and an estimated cost field. Confidence intervals are computed with bootstrap resampling; paired bootstrap comparison is available for two runs.

> The project intentionally does **not** ship fabricated experimental results. Run the benchmark and write the real numbers into your portfolio README.

## Windows 10/11 setup (PowerShell)

### 1. Install Python

Install Python 3.10+ from Python.org and tick **Add Python to PATH**.

Verify:

```powershell
python --version
```

### 2. Open the project

Open the extracted `agentprobe` folder in VS Code. Use the VS Code PowerShell terminal from the folder containing `run_eval.py`.

### 3. Create and activate the virtual environment

```powershell
python -m venv venv
venv\\Scripts\\Activate.ps1
```

If PowerShell blocks the activation script:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

### 4. Install dependencies

```powershell
pip install -r requirements.txt
```

### 5. Create `.env`

```powershell
copy .env.example .env
```

For the first run, keep:

```text
LLM_MODEL=mock
```

### 6. Run the full benchmark offline

```powershell
python run_eval.py --suite all --variant all --model mock --seeds 3 --fault 0.1 --guard block
```

The first start automatically creates `sandbox/sample.db`, safe fixture documents, database tables, and benchmark support data.

### 7. Run tests

```powershell
pytest
```

The included suite has more than 25 tests covering tools, validation, guardrails, metrics, failure classification, the harness, runner behavior, and optional modules.

### 8. Start the backend

```powershell
uvicorn agentprobe.api.main:app --reload --port 8000
```

Open:

`http://localhost:8000/docs`

### 9. Start the dashboard in a second terminal

Activate the virtual environment again:

```powershell
venv\\Scripts\\Activate.ps1
```

Then:

```powershell
streamlit run dashboard\\app.py
```

Open:

`http://localhost:8501`

## Useful demo flow

1. Run `react` with MockLLM, 1 seed, fault 0.0, guard `block`.
2. Run `plan_execute` with the same suite/seeds.
3. Run `react_reflect` with the same suite/seeds.
4. Compare runs on the Leaderboard page.
5. Open a failed episode in Trace Viewer and inspect each action/observation.
6. Run the safety suite with guard `off` and `block` and compare the safety trade-off.
7. Use Scientist-lite with a goal such as `reduce bad-argument errors`.
8. Export preference pairs for a run from the API endpoint.

## Real model configuration (optional)

Use an OpenAI-compatible provider by setting:

```text
LLM_BASE_URL=...
LLM_API_KEY=...
LLM_MODEL=...
```

`TOOL_MODE=native` uses native tool calling. Set `TOOL_MODE=json` for strict JSON action fallback if the provider/model does not support native tool calling.

The code path supports OpenAI-compatible endpoints such as OpenAI, Groq, Gemini-compatible endpoints, and Ollama, but provider model names can change. Use the exact model name accepted by the provider.

## Resume support

Runs store completed episodes in SQLite. Re-running a run with `--resume-run RUN_ID` skips episode identities that are already stored.

## Custom tasks

The Run Experiment page contains a Custom Task Builder. It writes complete task records to `tasks/custom.json`. Custom tasks can then be selected by their suite in subsequent evaluations.

## Optional interpretability

The core application does not need PyTorch or Transformers. To enable the optional module:

```powershell
pip install -r requirements-interp.txt
```

The dashboard remains usable when these packages are missing.

## Troubleshooting

- **`python` not recognized**: reinstall Python and enable Add Python to PATH.
- **Activation blocked**: run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`.
- **`streamlit` not recognized**: activate the virtual environment or use `python -m streamlit run dashboard\\app.py`.
- **`ModuleNotFoundError: agentprobe`**: run commands from the project root containing `run_eval.py`.
- **401 / invalid API key**: confirm `LLM_API_KEY` and `LLM_BASE_URL`, then restart the terminal.
- **429 rate limit**: use MockLLM/Ollama, reduce workers, or reduce seeds.
- **Model does not support tools**: use `TOOL_MODE=json` or select a tool-capable model.
- **Port already in use**: use another Uvicorn port and update `DASHBOARD_API_URL` in `.env`.
- **UnicodeEncodeError on Windows**: set `$env:PYTHONUTF8=1` before rerunning.
- **Torch installation fails**: skip `requirements-interp.txt`; the core project does not need it.

## Adding tools, tasks and variants

- Add a new Pydantic args model and `Tool` registration in `agentprobe/tools/builtin.py` or a dedicated tool module.
- Add tasks as complete JSON records in `tasks/*.json`.
- Add a variant to `agentprobe/agent/variants.py` and define its harness behavior in `agentprobe/agent/harness.py`.
- Add grader logic in `agentprobe/eval/graders.py` when a new evaluation style is necessary.

## Research reporting

The project specification expects the final portfolio version to contain:

- real experimental findings
- a screenshot of the Trace Viewer
- one failure case analyzed in depth
- an honest results section in this README
- a short technical note/blog post
- GitHub publication without secrets

Do not invent benchmark numbers.
