"""The deciders a bench run can compare: one registry entry per decision model, keyed by `DeciderName`.

An entry says how to build the model, how its use is paid (tokens, GPU time, or a local machine), what the run
manifest records about where it runs, and which environment variables it needs. A new decision model is one enum
member, one entry and its declared `ModelLimits` in the library.
"""

from __future__ import annotations

import os
import platform
import subprocess
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Literal

import anyio
import httpx2

from toolhunch.decision import OpenAILogprobModel, clef, clm, jev, strands_decider
from toolhunch_bench.structured import StructuredChoiceModel

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Mapping

    from toolhunch.decision import DecisionModel, DecisionRequest, DecisionResponse, ModelLimits, QuestionKind

__all__ = [
    "CLM_DEPLOYMENT",
    "CLM_MODEL",
    "DECIDERS",
    "JEV_MODEL",
    "LOGPROB_MODEL",
    "LUNA_MODEL",
    "Billing",
    "DeciderName",
    "DeciderSpec",
    "SerialDecisionModel",
    "decision_model",
    "local_provenance",
    "missing_env",
]


class DeciderName(StrEnum):
    """The deciders a run can compare. The values are the names runs record, so they never change."""

    JEV = "jev"
    LOGPROB = "logprob"
    LUNA = "luna"
    CLM = "clm"
    CLM_LOCAL = "clm-local"
    STRANDS = "strands"
    CLEF = "clef"
    CLEF_FLASH = "clef-flash"


type Billing = Literal["tokens", "gpu-time", "local"]
"""How a decider's use is paid: per token, per second of a rented GPU, or not at all (a local machine)."""

JEV_MODEL = "jev-1.13.0"
"""The pinned Jev version: `jev-latest` would move under a run."""
LOGPROB_MODEL = "gpt-4.1-mini-2025-04-14"
"""The logprob baseline's model: an OpenAI snapshot that returns 20 top logprobs."""
LUNA_MODEL = "gpt-6-luna"
"""GPT-6 Luna, reasoning off: the `luna` decider asks it for a structured letter, the direct agent arms for a call."""
CLM_MODEL = "clm-latest"
"""The model `clm-serve` answers with."""
CLM_DEPLOYMENT: dict[str, Any] = {
    "script": "bench/deploy/clm_modal.py",
    "vllm": "0.30.0",
    "contrastive_lm": "0.1.0",
    "gpu": "L4",
    "usd_per_gpu_hour": 0.80,
    "price_checked": "2026-09-28",
}
"""Where the `clm` model runs, and the list price of its GPU, which Modal credits pay."""

_STRANDS_URL = ("STRANDS_BASE_URL", "http://127.0.0.1:8000")
_CLM_LOCAL_URL = ("CLM_LOCAL_BASE_URL", "http://127.0.0.1:8700")
_CLM_ENCODER_HEALTH = "http://127.0.0.1:8090/health"
_CLOUDFLARE_ENV = ("CLOUDFLARE_ACCOUNT_ID", "CLOUDFLARE_API_KEY")
# The P2 probe (20261004T170651Z): 265 billed tokens for 180 heuristic ones at 5 options, 6,638 for 1,829 at 255,
# so about 19 more per option and none per request. Estimates only; the guard prices what each reply reports.
_CLEF_OPTION_OVERHEAD = 19
_CLEF_API_VERSION = "2026-10-01.epoch"


@dataclass(frozen=True, slots=True, kw_only=True)
class DeciderSpec:
    """One decider: how to build its model, how it is paid and what a run manifest records about it.

    Attributes:
        name: The decider's name in runs.
        label: Its name in reports and charts.
        provider: Who bills it in the cost ledger; `local` for a model on this machine.
        model: The pinned model id it asks.
        billing: How its use is paid.
        make: Builds the model with the given number of internal retries; reads the environment when called.
        required_env: The variables it needs, checked by name before a run; their values are never printed.
        provenance: Where it runs, with pinned versions; no secret, no account ID and no deployment URL.
        serial: Ask it one request at a time, for a server whose behaviour under concurrent requests is unverified.
        local_url: The variable naming a local server's root, and its default; `None` for a hosted model.
        request_overhead_tokens: Tokens it bills beyond a request's text, per request, for estimates only.
        option_overhead_tokens: Tokens it bills beyond a request's text, per choice option, for estimates only.
        estimate_output_tokens: Output tokens per ask, for estimates only.
    """

    name: DeciderName
    label: str
    provider: str
    model: str
    billing: Billing
    make: Callable[[int], DecisionModel]
    required_env: tuple[str, ...] = ()
    provenance: Mapping[str, Any] = field(default_factory=dict[str, Any])
    serial: bool = False
    local_url: tuple[str, str] | None = None
    request_overhead_tokens: int = 0
    option_overhead_tokens: int = 0
    estimate_output_tokens: int = 0


def _url(variable: tuple[str, str]) -> str:
    name, default = variable
    return os.environ.get(name) or default


DECIDERS: Mapping[DeciderName, DeciderSpec] = {
    spec.name: spec
    for spec in (
        DeciderSpec(
            name=DeciderName.JEV,
            label="Jev",
            provider="typesafe",
            model=JEV_MODEL,
            billing="tokens",
            make=lambda retries: jev(JEV_MODEL, max_retries=retries),
            required_env=("TYPESAFE_API_KEY",),
            provenance={"endpoint": "api.typesafe.ai"},
            request_overhead_tokens=300,
        ),
        DeciderSpec(
            name=DeciderName.LOGPROB,
            label="GPT-4.1 mini",
            provider="openai",
            model=LOGPROB_MODEL,
            billing="tokens",
            make=lambda retries: OpenAILogprobModel(LOGPROB_MODEL, max_retries=retries),
            required_env=("OPENAI_API_KEY",),
            provenance={"endpoint": "api.openai.com"},
            request_overhead_tokens=120,
            estimate_output_tokens=1,
        ),
        DeciderSpec(
            name=DeciderName.LUNA,
            label="GPT-6 Luna",
            provider="openai",
            model=LUNA_MODEL,
            billing="tokens",
            make=lambda retries: StructuredChoiceModel(LUNA_MODEL, max_retries=retries),
            required_env=("OPENAI_API_KEY",),
            provenance={"endpoint": "api.openai.com", "reasoning": "off"},
            request_overhead_tokens=120,
            estimate_output_tokens=16,
        ),
        DeciderSpec(
            name=DeciderName.CLM,
            label="CLM (Modal)",
            provider="modal",
            model=CLM_MODEL,
            billing="gpu-time",
            make=lambda retries: clm(os.environ["CLM_BASE_URL"], model=CLM_MODEL, max_retries=retries),
            required_env=("CLM_BASE_URL",),
            provenance=CLM_DEPLOYMENT,
        ),
        DeciderSpec(
            name=DeciderName.CLM_LOCAL,
            label="CLM",
            provider="local",
            model=CLM_MODEL,
            billing="local",
            make=lambda retries: clm(_url(_CLM_LOCAL_URL), model=CLM_MODEL, api_key_env=None, max_retries=retries),
            provenance={
                "script": "bench/deploy/clm_local/serve.sh",
                "contrastive_lm": "0.1.0",
                "head": "Contrastive-LM/CLM-v0.1-8B CLM_v0.1-8B.pt",
                "encoder": "Qwen/Qwen3-8B @ b968826, transformers, bf16, last-token pooling",
            },
            serial=True,
            local_url=_CLM_LOCAL_URL,
        ),
        DeciderSpec(
            name=DeciderName.STRANDS,
            label="Strands Decider 2B",
            provider="local",
            model="strands-decider-2B-hobson-v19",
            billing="local",
            make=lambda retries: strands_decider(_url(_STRANDS_URL), max_retries=retries),
            provenance={
                "script": "bench/deploy/strands_local.sh",
                "code": "strands-labs/strands-decider @ 75c9fd3",
                "checkpoint": "StrandsAgents/strands-decider-2B-hobson-v19 @ bb282d7",
                "base": "Qwen/Qwen3.5-2B-Base @ b1485b2",
                "flags": "--strict-window",
            },
            serial=True,
            local_url=_STRANDS_URL,
        ),
        DeciderSpec(
            name=DeciderName.CLEF,
            label="Clef",
            provider="cloudflare",
            model="clef",
            billing="tokens",
            make=lambda retries: clef("clef", max_retries=retries),
            required_env=_CLOUDFLARE_ENV,
            provenance={
                "endpoint": "api.cloudflare.com Workers AI @cf/cloudflare/clef",
                "version_pinned": False,
                "api_version_at_probe": _CLEF_API_VERSION,
            },
            option_overhead_tokens=_CLEF_OPTION_OVERHEAD,
        ),
        DeciderSpec(
            name=DeciderName.CLEF_FLASH,
            label="Clef-flash",
            provider="cloudflare",
            model="clef-flash",
            billing="tokens",
            make=lambda retries: clef("clef-flash", max_retries=retries),
            required_env=_CLOUDFLARE_ENV,
            provenance={
                "endpoint": "api.cloudflare.com Workers AI @cf/cloudflare/clef-flash",
                "version_pinned": False,
                "api_version_at_probe": _CLEF_API_VERSION,
            },
            option_overhead_tokens=_CLEF_OPTION_OVERHEAD,
        ),
    )
}
"""Every decider, by name."""


class SerialDecisionModel:
    """A decision model asked one request at a time: the decider asks a first round's questions concurrently."""

    def __init__(self, inner: DecisionModel) -> None:
        self._inner = inner
        self._lock = anyio.Lock()

    @property
    def model_id(self) -> str:
        """The wrapped model's identity."""
        return self._inner.model_id

    @property
    def limits(self) -> ModelLimits:
        """The wrapped model's limits."""
        return self._inner.limits

    @property
    def question_kinds(self) -> frozenset[QuestionKind]:
        """The wrapped model's question kinds."""
        return self._inner.question_kinds

    @property
    def prompt_version(self) -> str | None:
        """The wrapped model's prompt version."""
        return self._inner.prompt_version

    def __repr__(self) -> str:
        return repr(self._inner)

    async def ask(self, request: DecisionRequest, /, **options: Any) -> DecisionResponse:
        """Ask the wrapped model once no other ask is in flight."""
        async with self._lock:
            return await self._inner.ask(request, **options)

    async def aclose(self) -> None:
        """Close the wrapped model's client, if it has one."""
        if (close := getattr(self._inner, "aclose", None)) is not None:
            await close()


def decision_model(name: DeciderName, *, max_retries: int = 3) -> DecisionModel:
    """The model behind `name`, asked one request at a time when its entry says so. Nothing is contacted here."""
    spec = DECIDERS[name]
    model = spec.make(max_retries)
    return SerialDecisionModel(model) if spec.serial else model


def missing_env(names: Iterable[DeciderName]) -> list[str]:
    """The variables the deciders `names` need that are not set, by name only."""
    needed = dict.fromkeys(variable for name in names for variable in DECIDERS[name].required_env)
    return [variable for variable in needed if not os.environ.get(variable)]


async def local_provenance(name: DeciderName, *, client: httpx2.AsyncClient | None = None) -> dict[str, Any]:
    """What a run manifest records about `name`: its entry's provenance, and more for a local model.

    For a local model it adds its servers' `/health` answers and the machine it runs on. A health endpoint that
    does not answer is recorded as `None`: the run's first ask will fail if it is down.
    """
    spec = DECIDERS[name]
    record: dict[str, Any] = {"label": spec.label, "billing": spec.billing, **spec.provenance}
    if spec.local_url is None:
        return record
    urls = [f"{_url(spec.local_url).rstrip('/')}/health"]
    if name is DeciderName.CLM_LOCAL:
        urls.append(_CLM_ENCODER_HEALTH)
    http = client if client is not None else httpx2.AsyncClient(timeout=10.0)
    health: list[dict[str, Any] | None] = []
    try:
        for url in urls:
            try:
                response = await http.get(url)
                health.append(response.json() if response.is_success else None)
            except (httpx2.HTTPError, ValueError):
                health.append(None)
    finally:
        if client is None:
            await http.aclose()
    return record | {"health": health, "hardware": _hardware()}


def _hardware() -> dict[str, Any]:
    """The chip, memory and operating system of this machine, as a local decider's latency depends on them."""

    def sysctl(key: str) -> str | None:
        try:
            return subprocess.run(["sysctl", "-n", key], capture_output=True, text=True, check=True).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    memory = sysctl("hw.memsize")
    return {
        "chip": sysctl("machdep.cpu.brand_string") or platform.processor(),
        "memory_gb": round(int(memory) / 2**30) if memory and memory.isdigit() else None,
        "os": f"macOS {platform.mac_ver()[0]}" if platform.mac_ver()[0] else platform.platform(),
    }
