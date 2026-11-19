"""
dpo_training.py: DPO alignment for MedGemma-2B (save as gemma-2b-dpo/training.py)

Loads google/gemma-2b (4-bit), merges the trained SFT LoRA into it, attaches a fresh LoRA
and trains with DPOTrainer. The reference model is the same network with the new LoRA
disabled (= the merged SFT model), so no second model needs to be loaded.

Quick test:
    python training.py --sft_adapter /content/model/lora_adapter --output_dir /content/model/dpo_test \
        --max_train_samples 200 --max_eval_samples 50 --max_length 512 --push_to_hub False

Real run:
    python training.py --sft_adapter /content/model/lora_adapter --output_dir /content/model/dpo_adapter \
        --num_train_epochs 1 --push_to_hub True --hub_model_id anmolsehgal/medgemma-2b-dpo-v2
"""

import gc
import inspect
import logging
import os
import sys
from dataclasses import dataclass, field
from typing import List, Optional

import torch
from datasets import load_dataset
from huggingface_hub import get_token, login
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    set_seed,
)
from trl import DPOConfig, DPOTrainer

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


# ── Config ─────────────────────────────────────────────────────────────────────
@dataclass
class DPODataConfig:
    dataset_id: str = field(
        default="WhiteF4lcon/medical-rlhf-pairs",
        metadata={"help": "HF dataset with prompt / chosen / rejected style columns"},
    )
    val_split: float = field(default=0.10)
    seed: int = field(default=42)
    max_train_samples: int = field(default=0, metadata={"help": "0 = use all"})
    max_eval_samples: int = field(default=200)


@dataclass
class DPOModelConfig:
    model_id: str = field(default="google/gemma-2b")
    sft_adapter: str = field(default="model/lora_adapter", metadata={"help": "Path or hub id of the SFT LoRA adapter"})
    output_dir: str = field(default="model/dpo_adapter")
    lora_r: int = field(default=16)
    lora_alpha: int = field(default=32)
    lora_dropout: float = field(default=0.1)
    target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    use_rslora: bool = field(default=False)
    num_train_epochs: int = field(default=1)
    max_steps: int = field(default=-1, metadata={"help": "Overrides epochs if > 0"})
    learning_rate: float = field(default=1e-5)
    beta: float = field(default=0.5)
    loss_type: str = field(default="ipo", metadata={"help": "sigmoid | ipo | hinge | ..."})
    max_length: int = field(default=1024)
    max_prompt_length: int = field(default=512)
    per_device_train_batch_size: int = field(default=1)
    gradient_accumulation_steps: int = field(default=8)
    eval_steps: int = field(default=100)
    push_to_hub: bool = field(default=False)
    hub_model_id: Optional[str] = field(default=None)


# ── Helpers ────────────────────────────────────────────────────────────────────
def _use_bf16() -> bool:
    return torch.cuda.is_available() and torch.cuda.is_bf16_supported()


def _filter_kwargs(cls, kwargs: dict) -> dict:
    """Drop keys the installed version of `cls` does not accept."""
    params = inspect.signature(cls.__init__).parameters
    ok = {k: v for k, v in kwargs.items() if k in params}
    dropped = sorted(set(kwargs) - set(ok))
    if dropped:
        logger.warning("%s: ignoring unsupported args: %s", cls.__name__, dropped)
    return ok


# ── Model loading ──────────────────────────────────────────────────────────────
def load_policy_model(cfg: DPOModelConfig):
    """Base (4-bit) -> merge SFT LoRA -> fresh DPO LoRA."""
    compute_dtype = torch.bfloat16 if _use_bf16() else torch.float16
    logger.info("Compute dtype: %s", compute_dtype)

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
    )
    load_kwargs = dict(quantization_config=bnb, device_map="auto")

    logger.info("Loading base model: %s", cfg.model_id)
    try:
        base = AutoModelForCausalLM.from_pretrained(cfg.model_id, dtype=compute_dtype, **load_kwargs)
    except (TypeError, ValueError):
        base = AutoModelForCausalLM.from_pretrained(cfg.model_id, torch_dtype=compute_dtype, **load_kwargs)

    logger.info("Attaching and merging SFT adapter: %s", cfg.sft_adapter)
    merged = PeftModel.from_pretrained(base, cfg.sft_adapter).merge_and_unload()
    merged.config.use_cache = False

    merged = prepare_model_for_kbit_training(merged)
    lora_kwargs = dict(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=cfg.target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )
    if cfg.use_rslora:
        lora_kwargs["use_rslora"] = True
    model = get_peft_model(merged, LoraConfig(**lora_kwargs))
    model.print_trainable_parameters()
    return model


# ── Dataset ────────────────────────────────────────────────────────────────────
PROMPT_COLS = ["instruction", "prompt", "question", "input"]
CHOSEN_COLS = ["positive", "chosen", "accepted"]
REJECTED_COLS = ["negative", "rejected"]


def _pick(cols, candidates):
    return next((c for c in candidates if c in cols), None)


def build_dpo_dataset(cfg: DPODataConfig):
    logger.info("Loading DPO dataset: %s", cfg.dataset_id)
    raw = load_dataset(cfg.dataset_id)
    base = raw["train"] if "train" in raw else raw[list(raw.keys())[0]]
    cols = base.column_names

    p_col, c_col, r_col = _pick(cols, PROMPT_COLS), _pick(cols, CHOSEN_COLS), _pick(cols, REJECTED_COLS)
    if not (p_col and c_col and r_col):
        raise ValueError(f"Could not find prompt/chosen/rejected columns in {cols}")
    logger.info("Using columns: prompt=%s chosen=%s rejected=%s", p_col, c_col, r_col)

    def fmt(x):
        return {
            "prompt": str(x[p_col]).strip(),
            "chosen": str(x[c_col]).strip(),
            "rejected": str(x[r_col]).strip(),
        }

    ds = base.map(fmt, remove_columns=cols)
    ds = ds.filter(lambda x: x["prompt"] and x["chosen"] and x["rejected"] and x["chosen"] != x["rejected"])

    split = ds.train_test_split(test_size=cfg.val_split, seed=cfg.seed)
    train_ds, val_ds = split["train"], split["test"]
    if cfg.max_train_samples:
        train_ds = train_ds.select(range(min(cfg.max_train_samples, len(train_ds))))
    val_ds = val_ds.select(range(min(cfg.max_eval_samples, len(val_ds))))

    logger.info("DPO split -> train: %s | val: %s", f"{len(train_ds):,}", f"{len(val_ds):,}")
    logger.info("Sample prompt: %s", train_ds[0]["prompt"][:120])
    return train_ds, val_ds


# ── Training ───────────────────────────────────────────────────────────────────
def train_dpo(model, tokenizer, train_ds, val_ds, cfg: DPOModelConfig) -> None:
    bf16 = _use_bf16()
    params = inspect.signature(DPOConfig.__init__).parameters

    args = dict(
        output_dir=cfg.output_dir.rstrip("/") + "_ckpt",  # checkpoints kept apart from the final adapter
        beta=cfg.beta,
        loss_type=cfg.loss_type,
        learning_rate=cfg.learning_rate,
        num_train_epochs=cfg.num_train_epochs,
        max_steps=cfg.max_steps,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        lr_scheduler_type="cosine",
        weight_decay=0.01,
        bf16=bf16,
        fp16=not bf16,
        logging_steps=10,
        save_steps=cfg.eval_steps,
        eval_steps=cfg.eval_steps,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        report_to="none",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="paged_adamw_8bit",
        max_grad_norm=0.3,
        max_length=cfg.max_length,
        max_prompt_length=cfg.max_prompt_length,
        push_to_hub=False,  # adapter is uploaded manually at the end
    )
    args["eval_strategy" if "eval_strategy" in params else "evaluation_strategy"] = "steps"
    if "warmup_ratio" in params:
        args["warmup_ratio"] = 0.1
    else:
        args["warmup_steps"] = 0.1

    dpo_cfg = DPOConfig(**_filter_kwargs(DPOConfig, args))

    trainer_kwargs = dict(
        model=model,
        ref_model=None,  # PEFT model: reference = same network with the DPO LoRA disabled (= merged SFT)
        args=dpo_cfg,
        train_dataset=train_ds,
        eval_dataset=val_ds,
    )
    t_params = inspect.signature(DPOTrainer.__init__).parameters
    trainer_kwargs["processing_class" if "processing_class" in t_params else "tokenizer"] = tokenizer
    trainer = DPOTrainer(**trainer_kwargs)

    logger.info("Starting DPO training (loss=%s, beta=%s)...", cfg.loss_type, cfg.beta)
    trainer.train()

    os.makedirs(cfg.output_dir, exist_ok=True)
    trainer.save_model(cfg.output_dir)
    tokenizer.save_pretrained(cfg.output_dir)
    logger.info("DPO adapter saved -> %s", cfg.output_dir)

    if cfg.push_to_hub and cfg.hub_model_id:
        from huggingface_hub import HfApi

        api = HfApi()
        api.create_repo(repo_id=cfg.hub_model_id, repo_type="model", exist_ok=True)
        api.upload_folder(folder_path=cfg.output_dir, repo_id=cfg.hub_model_id, repo_type="model")
        logger.info("Model pushed to HuggingFace -> %s", cfg.hub_model_id)

    del trainer
    gc.collect()
    torch.cuda.empty_cache()


# ── Entrypoint ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    hf_token = os.environ.get("HF_TOKEN") or get_token()
    if not hf_token:
        raise EnvironmentError("No Hugging Face token found. Set HF_TOKEN or run huggingface_hub.login().")
    login(token=hf_token)

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    logger.info("Using CUDA device: %s", os.environ["CUDA_VISIBLE_DEVICES"])

    parser = HfArgumentParser((DPODataConfig, DPOModelConfig))
    data_cfg, model_cfg = parser.parse_args_into_dataclasses()
    set_seed(data_cfg.seed)

    tokenizer = AutoTokenizer.from_pretrained(model_cfg.model_id, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    train_ds, val_ds = build_dpo_dataset(data_cfg)
    model = load_policy_model(model_cfg)
    train_dpo(model, tokenizer, train_ds, val_ds, model_cfg)

    del model
    gc.collect()
    torch.cuda.empty_cache()
    logger.info("Done.")