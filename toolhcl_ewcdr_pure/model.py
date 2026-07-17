from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from torch import nn

from .utils import EXPECTED_OLD_TOOL_COUNTS, EXPECTED_TOOL_COUNTS


class FrozenLlamaEncoder(nn.Module):
    def __init__(self, backbone: nn.Module, tokenizer: Any, max_length: int):
        super().__init__()
        self.backbone = backbone
        self.tokenizer = tokenizer
        self.max_length = int(max_length)
        self.hidden_size = int(backbone.config.hidden_size)
        self.backbone.eval()
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False

    @torch.no_grad()
    def forward(self, texts: Sequence[str]) -> torch.Tensor:
        device = next(self.backbone.parameters()).device
        encoded = self.tokenizer(
            list(texts),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        ).to(device)
        outputs = self.backbone(
            input_ids=encoded.input_ids,
            attention_mask=encoded.attention_mask,
            return_dict=True,
            use_cache=False,
        )
        last_index = encoded.attention_mask.long().sum(dim=1) - 1
        rows = torch.arange(encoded.input_ids.shape[0], device=device)
        return outputs.last_hidden_state[rows, last_index].detach()


class QueryProjection(nn.Module):
    def __init__(self, hidden_size: int = 4096, projection_hidden: int = 1024, output_size: int = 384, dropout: float = 0.1):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(hidden_size, projection_hidden),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(projection_hidden, output_size),
            nn.LayerNorm(output_size),
        )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.layers(hidden.float())


class PureEwcDrRetriever(nn.Module):
    def __init__(
        self,
        *,
        num_tools: int,
        encoder: FrozenLlamaEncoder | None,
        hidden_size: int = 4096,
        projection_hidden: int = 1024,
        query_size: int = 384,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.encoder = encoder
        self.query_projection = QueryProjection(hidden_size, projection_hidden, query_size, dropout)
        self.classifier = nn.Linear(query_size, int(num_tools))
        self.num_tools = int(num_tools)

    def encode(self, texts: Sequence[str]) -> torch.Tensor:
        if self.encoder is None:
            raise RuntimeError("Text forward requires an attached frozen encoder")
        return self.encoder(texts)

    def forward_hidden(self, query_hidden: torch.Tensor) -> torch.Tensor:
        query_vector = self.query_projection(query_hidden)
        return self.classifier(query_vector)

    def forward_text(self, texts: Sequence[str]) -> torch.Tensor:
        return self.forward_hidden(self.encode(texts))

    def forward_batch(self, batch: Mapping[str, Any], device: torch.device) -> torch.Tensor:
        if "query_hidden" in batch:
            hidden = batch["query_hidden"].to(device, non_blocking=True)
            return self.forward_hidden(hidden)
        return self.forward_text(batch["query_text"])


def load_frozen_encoder(config: Mapping[str, Any], device: torch.device) -> FrozenLlamaEncoder:
    from transformers import AutoModel, AutoTokenizer

    model_config = config["model"]
    encoder_path = model_config["encoder_path"]
    tokenizer_path = model_config.get("tokenizer_path") or encoder_path
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    dtype_name = str(model_config.get("encoder_dtype", "bfloat16")).lower()
    dtype = torch.bfloat16 if dtype_name == "bfloat16" else torch.float16 if dtype_name == "float16" else torch.float32
    kwargs: dict[str, Any] = {
        "torch_dtype": dtype,
        "low_cpu_mem_usage": True,
        "local_files_only": True,
    }
    if device.type == "cuda" and bool(model_config.get("load_direct_to_device", True)):
        kwargs["device_map"] = {"": device.index if device.index is not None else 0}
    backbone = AutoModel.from_pretrained(encoder_path, **kwargs)
    if "device_map" not in kwargs:
        backbone = backbone.to(device)
    return FrozenLlamaEncoder(backbone, tokenizer, int(model_config.get("max_length", 512)))


def build_retriever(
    config: Mapping[str, Any], stage: str, *, encoder: FrozenLlamaEncoder | None = None
) -> PureEwcDrRetriever:
    if stage not in EXPECTED_TOOL_COUNTS:
        raise ValueError(f"Unknown stage: {stage}")
    model_config = config["model"]
    model = PureEwcDrRetriever(
        num_tools=EXPECTED_TOOL_COUNTS[stage],
        encoder=encoder,
        hidden_size=int(model_config.get("hidden_size", 4096)),
        projection_hidden=int(model_config.get("projection_hidden", 1024)),
        query_size=int(model_config.get("query_size", 384)),
        dropout=float(model_config.get("dropout", 0.1)),
    )
    for name, parameter in model.named_parameters():
        parameter.requires_grad = not name.startswith("encoder.")
    return model


def named_trainable_parameters(model: nn.Module) -> dict[str, nn.Parameter]:
    return {name: parameter for name, parameter in model.named_parameters() if parameter.requires_grad}


def trainable_summary(model: nn.Module) -> dict[str, Any]:
    values = named_trainable_parameters(model)
    return {"names": list(values), "parameter_count": sum(int(value.numel()) for value in values.values())}


def initialize_from_previous(
    model: PureEwcDrRetriever, checkpoint_path: str | Path, stage: str, *, verify_exact: bool = True
) -> dict[str, Any]:
    if stage not in EXPECTED_OLD_TOOL_COUNTS:
        raise ValueError(f"No previous-stage expansion is defined for {stage}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    old_count = int(checkpoint["num_tools"])
    expected_old = EXPECTED_OLD_TOOL_COUNTS[stage]
    assert old_count == expected_old, f"{stage} old_num_tools={old_count}, expected {expected_old}"
    assert model.num_tools == EXPECTED_TOOL_COUNTS[stage]
    model.query_projection.load_state_dict(checkpoint["query_projection"], strict=True)
    old_weight = checkpoint["classifier"]["weight"]
    old_bias = checkpoint["classifier"]["bias"]
    with torch.no_grad():
        model.classifier.weight[:old_count].copy_(old_weight)
        model.classifier.bias[:old_count].copy_(old_bias)
    if verify_exact:
        torch.testing.assert_close(model.classifier.weight[:old_count].cpu(), old_weight, rtol=0, atol=0)
        torch.testing.assert_close(model.classifier.bias[:old_count].cpu(), old_bias, rtol=0, atol=0)
    return checkpoint


def checkpoint_payload(
    model: PureEwcDrRetriever,
    *,
    stage: str,
    epoch: int,
    training_history: Sequence[Mapping[str, Any]],
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "format": "toolhcl_pure_ewcdr_v1",
        "stage": stage,
        "epoch": int(epoch),
        "num_tools": model.num_tools,
        "query_projection": {k: v.detach().cpu() for k, v in model.query_projection.state_dict().items()},
        "classifier": {k: v.detach().cpu() for k, v in model.classifier.state_dict().items()},
        "training_history": [dict(row) for row in training_history],
        "metadata": dict(metadata),
    }


def save_checkpoint(path: str | Path, model: PureEwcDrRetriever, **kwargs: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint_payload(model, **kwargs), target)


def load_checkpoint(model: PureEwcDrRetriever, path: str | Path) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if int(checkpoint["num_tools"]) != model.num_tools:
        raise AssertionError(f"Checkpoint has {checkpoint['num_tools']} tools, model has {model.num_tools}")
    model.query_projection.load_state_dict(checkpoint["query_projection"], strict=True)
    model.classifier.load_state_dict(checkpoint["classifier"], strict=True)
    return checkpoint
