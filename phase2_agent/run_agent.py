from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from phase2_agent.config import AgentSettings
from phase2_agent.optimizer import LoRAOptimizationAgent


def main() -> int:
    settings = AgentSettings.from_repo_root(REPO_ROOT)
    agent = LoRAOptimizationAgent(settings)
    code = agent.run()
    settings.summary_path.write_text(json.dumps(agent.summary(), indent=2), encoding="utf-8")
    print(json.dumps(agent.summary(), indent=2))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
