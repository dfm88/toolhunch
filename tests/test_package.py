from importlib.metadata import version

import toolhunch
import toolhunch.decision
from toolhunch.decision import base, decider, jev_wire, logprobs, planner


def test_version_matches_distribution_metadata() -> None:
    assert toolhunch.__version__ == version("toolhunch")


def test_the_decision_stage_is_exported_at_the_top_level() -> None:
    names = [
        "Abstention",
        "ChoiceDecider",
        "Decider",
        "Decision",
        "DecisionError",
        "DecisionModel",
        "DecisionUsage",
        "JevWireModel",
        "ModelLimits",
        "OpenAILogprobModel",
        "ThresholdKey",
        "clm",
        "jev",
    ]
    assert set(names) <= set(toolhunch.__all__)
    for name in names:
        assert getattr(toolhunch, name) is getattr(toolhunch.decision, name)


def test_the_decision_package_exports_what_its_modules_do() -> None:
    # The logprob adapter's prompt version and system prompt stay in `toolhunch.decision.logprobs` (the adapter
    # exposes the version as `OpenAILogprobModel.prompt_version`): a flat `toolhunch.decision.PROMPT_VERSION` would
    # read as the planner's decision prompt version. The planner's own names are not re-exported either, except
    # the error a caller of the decider catches.
    kept_in_their_module = {"PROMPT_VERSION", "SYSTEM_PROMPT"}
    modules = (base, jev_wire, logprobs, decider)
    public = {name for module in modules for name in module.__all__}

    assert set(toolhunch.decision.__all__) == (public - kept_in_their_module) | {"CandidatesDoNotFit"}
    assert toolhunch.decision.CandidatesDoNotFit is planner.CandidatesDoNotFit
    assert kept_in_their_module <= set(logprobs.__all__)
    for module in modules:
        for name in module.__all__:
            if name not in kept_in_their_module:
                assert getattr(toolhunch.decision, name) is getattr(module, name)
