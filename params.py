from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


@dataclass
class ModelConfig:
    """Config for the language model (shared across all benchmarks)."""

    # Model identification
    name: str = "google/gemma-3-12b-it"
    backend: str = "vllm-service"

    # vllm-service: URL of the vLLM server (e.g. "http://localhost:8000/v1")
    api_base: Optional[str] = None

    # Generation parameters - used by both backends
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = -1
    max_tokens: int = 512
    seed: Optional[int] = None

    # Structured output support
    use_structured: bool = True

    # Chat template — whether the model's template has native tool-call roles.
    chat_template: bool = False
    template_has_tool_token: bool = False

    post_processed: Optional[bool] = False
    # Parser used to interpret model outputs: "json" or "pythonic"
    parser_type: str = "json"

    # Model in the loop
    model_in_the_loop: bool = False

    # Batch processing
    batch_size: int = 32


# Tool Calling Configs
@dataclass
class ToolCallingDataConfig:
    """Config for tool calling dataset."""

    path: str = "dataset/tool_calling/financial_dataset.json"
    format: str = "json"
    max_samples: Optional[int] = None
    shuffle: bool = False
    seed: int = 42


@dataclass
class ToolConfig:
    """Configuration for tools."""

    schema_path: Optional[str] = "prompts/financial_tools_schema.json"
    tools: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class ToolCallingPromptConfig:
    """Config for tool calling prompts."""

    # List of variable names from prompts/prompts.py to include
    sections: List[str] = field(
        default_factory=lambda: [
            "role_prompt",
            "tools_prompt",
            "output_format_prompt",
            "categories_prompt",
            "guardrails_prompt",
        ]
    )
    stop_sequences: List[str] = field(
        default_factory=lambda: [
            "\n\nUser Query:",
            "\n\n```",
            "```python",
            "\n\nUser:",
            "<|eot_id|>",
        ]
    )


@dataclass
class ToolCallingScoringConfig:
    """Config for tool calling scoring metrics."""

    metrics: List[str] = field(
        default_factory=lambda: [
            "when2call",
            "schema_reliability",
            "tool_pickup_and_hallucination",
            "variable_parsing_and_hallucination",
            "exact_match",
        ]
    )


@dataclass
class ToolCallingBenchmarkConfig:
    """Config for tool calling benchmark."""

    enabled: bool = True
    description: str = "Evaluate model's ability to use tools"
    data: ToolCallingDataConfig = field(default_factory=ToolCallingDataConfig)
    tool: ToolConfig = field(default_factory=ToolConfig)
    prompt: ToolCallingPromptConfig = field(default_factory=ToolCallingPromptConfig)
    scoring: ToolCallingScoringConfig = field(default_factory=ToolCallingScoringConfig)
    continue_on_error: bool = True
    log_errors: bool = True


# Guardrailing Configs
@dataclass
class GuardrailingDataConfig:
    """Config for guardrailing dataset."""

    path: str = "dataset/guardrailing/guardrails_data.jsonl"
    format: str = "jsonl"
    field_prompt: str = "prompt"
    field_label: str = "prompt_label"
    field_violated_categories: str = "violated_categories"
    max_samples: Optional[int] = None
    shuffle: bool = False
    seed: int = 42


@dataclass
class GuardrailingPromptConfig:
    """Config for guardrailing prompts."""

    instructions_path: str = "evaluator/guardrailing/instructions.txt"
    valid_labels: List[str] = field(default_factory=lambda: ["safe", "unsafe"])


@dataclass
class GuardrailingScoringConfig:
    """Config for guardrailing scoring metrics."""

    metrics: List[str] = field(
        default_factory=lambda: [
            "accuracy",
            "precision",
            "recall",
            "f1",
        ]
    )
    negative_label: str = "safe"


@dataclass
class GuardrailingBenchmarkConfig:
    """Config for guardrailing benchmark."""

    enabled: bool = True
    description: str = "Evaluate model's safety classification capabilities"
    data: GuardrailingDataConfig = field(default_factory=GuardrailingDataConfig)
    prompt: GuardrailingPromptConfig = field(default_factory=GuardrailingPromptConfig)
    scoring: GuardrailingScoringConfig = field(
        default_factory=GuardrailingScoringConfig
    )
    continue_on_error: bool = True
    log_errors: bool = True


@dataclass
class ConversationalScoringConfig:
    """Configuration for conversational content scoring."""

    # Non-LLM metrics (always computed)
    metrics: List[str] = field(
        default_factory=lambda: [
            "content_levenshtein_ratio",
            "content_rouge1",
            "content_rougeL",
            "content_bert_precision",
            "content_bert_recall",
            "content_bert_f1",
        ]
    )

    # LLM-as-judge metrics (only if judge enabled)
    llm_judge_metrics: List[str] = field(
        default_factory=lambda: [
            "content_similarity",
            "content_politeness",
            "topic_accuracy",
            "content_helpfulness",
        ]
    )


@dataclass
class LLMJudgePrompt:
    """Single LLM judge prompt configuration."""

    description: str
    system_prompt: str
    user_prompt: str
    output_type: str  # "score_1_5", "binary", or "text"
    metric_name: str


@dataclass
class JudgeConfig:
    """Global judge model configuration (shared across benchmarks)."""

    enabled: bool = True

    model: str = "gpt-4o-mini"  # OpenAI API models only

    temperature: float = 0.0
    max_tokens: int = 10
    batch_size: int = 32

    # Prompts configuration
    prompts_path: str = "config/llm_judge_prompts.yaml"
    prompts: Dict[str, LLMJudgePrompt] = field(default_factory=dict)

    # Runtime: store initialized judge model
    judge_model: Any = field(default=None, init=False, repr=False)

    def load_prompts(self) -> None:
        """Load judge prompts from YAML file."""
        prompts_file = Path(self.prompts_path)
        if not prompts_file.exists():
            raise FileNotFoundError(
                f"Judge prompts file not found: {self.prompts_path}"
            )

        with open(prompts_file, "r") as f:
            prompts_config = yaml.safe_load(f)

        for prompt_name, prompt_data in prompts_config.get("prompts", {}).items():
            self.prompts[prompt_name] = LLMJudgePrompt(
                description=prompt_data["description"],
                system_prompt=prompt_data["system_prompt"],
                user_prompt=prompt_data["user_prompt"],
                output_type=prompt_data["output_type"],
                metric_name=prompt_data["metric_name"],
            )


@dataclass
class ConversationalConfig:
    """Config for conversational content quality evaluation."""

    enabled: bool = True
    description: str = (
        "Evaluate conversational content quality (runs with tool calling)"
    )

    scoring: ConversationalScoringConfig = field(
        default_factory=ConversationalScoringConfig
    )


@dataclass
class EvalConfig:
    """Main evaluation configuration."""

    # Model config (shared)
    model: ModelConfig = field(default_factory=ModelConfig)

    # Benchmark configs
    tool_calling: ToolCallingBenchmarkConfig = field(
        default_factory=ToolCallingBenchmarkConfig
    )
    guardrailing: GuardrailingBenchmarkConfig = field(
        default_factory=GuardrailingBenchmarkConfig
    )
    conversational: ConversationalConfig = field(default_factory=ConversationalConfig)

    judge: JudgeConfig = field(default_factory=JudgeConfig)

    output_dir: str = "outputs/eval"
    save_predictions: bool = True
    save_format: str = "json"
    include_prompts: bool = True
    include_scores: bool = True
    include_raw_outputs: bool = True

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "EvalConfig":
        """Load config from unified YAML."""
        with open(yaml_path, "r") as f:
            config_dict = yaml.safe_load(f)

        # Build model config
        model_config = ModelConfig(**config_dict.get("model", {}))

        # Build benchmark configs
        benchmarks_dict = config_dict.get("benchmarks", {})

        # Tool calling
        tc_dict = benchmarks_dict.get("tool_calling", {})
        tool_calling_config = ToolCallingBenchmarkConfig(
            enabled=tc_dict.get("enabled", True),
            description=tc_dict.get("description", ""),
            data=ToolCallingDataConfig(**tc_dict.get("data", {})),
            tool=ToolConfig(**tc_dict.get("tool", {})),
            prompt=ToolCallingPromptConfig(**tc_dict.get("prompt", {})),
            scoring=ToolCallingScoringConfig(**tc_dict.get("scoring", {})),
            continue_on_error=tc_dict.get("continue_on_error", True),
            log_errors=tc_dict.get("log_errors", True),
        )

        # Guardrailing
        gr_dict = benchmarks_dict.get("guardrailing", {})
        guardrailing_config = GuardrailingBenchmarkConfig(
            enabled=gr_dict.get("enabled", True),
            description=gr_dict.get("description", ""),
            data=GuardrailingDataConfig(**gr_dict.get("data", {})),
            prompt=GuardrailingPromptConfig(**gr_dict.get("prompt", {})),
            scoring=GuardrailingScoringConfig(**gr_dict.get("scoring", {})),
            continue_on_error=gr_dict.get("continue_on_error", True),
            log_errors=gr_dict.get("log_errors", True),
        )

        # Conversational
        conv_dict = benchmarks_dict.get("conversational", {})
        conversational_config = ConversationalConfig(
            enabled=conv_dict.get("enabled", True),
            description=conv_dict.get("description", ""),
            scoring=ConversationalScoringConfig(**conv_dict.get("scoring", {})),
        )

        # Load judge config (global)
        judge_dict = config_dict.get("judge", {})
        judge_config = JudgeConfig(**judge_dict)

        # Load judge prompts if enabled
        if judge_config.enabled:
            judge_config.load_prompts()

        return cls(
            model=model_config,
            tool_calling=tool_calling_config,
            guardrailing=guardrailing_config,
            conversational=conversational_config,
            judge=judge_config,
            output_dir=config_dict.get("output_dir", "outputs/eval"),
            save_predictions=config_dict.get("save_predictions", True),
            save_format=config_dict.get("save_format", "json"),
            include_prompts=config_dict.get("include_prompts", True),
            include_scores=config_dict.get("include_scores", True),
            include_raw_outputs=config_dict.get("include_raw_outputs", True),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary for saving/logging."""
        return {
            "model": self.model.__dict__,
            "benchmarks": {
                "tool_calling": {
                    "enabled": self.tool_calling.enabled,
                    "description": self.tool_calling.description,
                    "data": self.tool_calling.data.__dict__,
                    "tool": self.tool_calling.tool.__dict__,
                    "prompt": self.tool_calling.prompt.__dict__,
                    "scoring": self.tool_calling.scoring.__dict__,
                    "continue_on_error": self.tool_calling.continue_on_error,
                    "log_errors": self.tool_calling.log_errors,
                },
                "guardrailing": {
                    "enabled": self.guardrailing.enabled,
                    "description": self.guardrailing.description,
                    "data": self.guardrailing.data.__dict__,
                    "prompt": self.guardrailing.prompt.__dict__,
                    "scoring": self.guardrailing.scoring.__dict__,
                    "continue_on_error": self.guardrailing.continue_on_error,
                    "log_errors": self.guardrailing.log_errors,
                },
                "conversational": {
                    "enabled": self.conversational.enabled,
                    "description": self.conversational.description,
                    "scoring": self.conversational.scoring.__dict__,
                },
            },
            "judge": {
                "enabled": self.judge.enabled,
                "model": self.judge.model,
                "temperature": self.judge.temperature,
                "max_tokens": self.judge.max_tokens,
                "batch_size": self.judge.batch_size,
                "prompts_path": self.judge.prompts_path,
            },
            "output_dir": self.output_dir,
            "save_predictions": self.save_predictions,
            "save_format": self.save_format,
            "include_prompts": self.include_prompts,
            "include_scores": self.include_scores,
            "include_raw_outputs": self.include_raw_outputs,
        }
