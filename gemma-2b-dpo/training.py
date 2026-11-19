"""
dpo_train.py — Production-ready DPO alignment for MedGemma-2B
Loads SFT adapter, merges it into base, applies fresh LoRA, trains with DPOTrainer.
"""

import gc
import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Optional

import torch
from datasets import load_dataset
from huggingface_hub import login
from peft import (
    LoraConfig,
    PeftModel,
    get_peft_model,
    prepare_model_for_kbit_training,
)
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
        metadata={"help": "HuggingFace dataset with instruction/positive/negative columns"},
    )
    val_split: float = field(default=0.10, metadata={"help": "Fraction of data held out for validation"})
    seed: int = field(default=42)
    use_generated_pairs: bool = field(
        default=False,
        metadata={"help": "Generate additional preference pairs using the SFT model"},
    )
    num_generated_pairs: int = field(
        default=2000,
        metadata={"help": "Number of AI-generated preference pairs to add"},
    )


@dataclass
class DPOModelConfig:
    model_id: str = field(default="google/gemma-2b")
    sft_adapter: str = field(default="model/lora_adapter", metadata={"help": "Path to trained SFT LoRA adapter"})
    output_dir: str = field(default="model/dpo_adapter")
    lora_r: int = field(default=64, metadata={"help": "LoRA rank"})
    lora_alpha: int = field(default=128, metadata={"help": "LoRA alpha"})
    lora_dropout: float = field(default=0.1, metadata={"help": "LoRA dropout"})
    target_modules: list = field(
        default_factory=lambda: ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        metadata={"help": "Target modules for LoRA"},
    )
    use_rslora: bool = field(default=True, metadata={"help": "Use rank-scaled LoRA"})
    push_to_hub: bool = field(default=False)
    hub_model_id: Optional[str] = field(default=None)


def get_bnb_config() -> BitsAndBytesConfig:
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )


# ── Model loading ──────────────────────────────────────────────────────────────
def load_and_merge_sft(model_cfg: DPOModelConfig) -> AutoModelForCausalLM:
    """Load base model, attach SFT adapter, merge weights, return plain model."""
    bnb = get_bnb_config()

    logger.info("Loading base model: %s", model_cfg.model_id)
    base = AutoModelForCausalLM.from_pretrained(
        model_cfg.model_id,
        quantization_config=bnb,
        device_map="auto",
        torch_dtype=torch.float16,
    )

    logger.info("Attaching SFT adapter: %s", model_cfg.sft_adapter)
    sft = PeftModel.from_pretrained(base, model_cfg.sft_adapter)

    logger.info("Merging SFT LoRA into base weights...")
    merged = sft.merge_and_unload()
    merged.config.use_cache = False
    logger.info("Merge complete")
    return merged


def apply_dpo_lora(merged_model: AutoModelForCausalLM, model_cfg: DPOModelConfig):
    """Prepare merged model for k-bit training and attach a fresh DPO LoRA."""
    merged_model = prepare_model_for_kbit_training(merged_model)

    lora_kwargs = dict(
        r=model_cfg.lora_r,
        lora_alpha=model_cfg.lora_alpha,
        lora_dropout=model_cfg.lora_dropout,
        target_modules=model_cfg.target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )
    if model_cfg.use_rslora:
        lora_kwargs["use_rslora"] = True

    lora_cfg = LoraConfig(**lora_kwargs)
    model = get_peft_model(merged_model, lora_cfg)
    model.print_trainable_parameters()
    logger.info("Fresh DPO LoRA applied")
    return model


def load_ref_model(model_cfg: DPOModelConfig) -> AutoModelForCausalLM:
    """Load a frozen reference model (base + SFT adapter, not merged) for KL penalty."""
    bnb = get_bnb_config()
    ref = AutoModelForCausalLM.from_pretrained(
        model_cfg.model_id,
        quantization_config=bnb,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    ref.load_adapter(model_cfg.sft_adapter)
    ref.eval()
    logger.info("Reference model loaded")
    return ref


# ── Dataset ────────────────────────────────────────────────────────────────────
def build_dpo_dataset(data_cfg: DPODataConfig):
    logger.info("Loading DPO dataset: %s", data_cfg.dataset_id)
    raw = load_dataset(data_cfg.dataset_id)

    def format_dpo(example):
        return {
            "prompt":   example["instruction"],
            "chosen":   example["positive"],
            "rejected": example["negative"],
        }

    dpo_ds = raw["train"].map(format_dpo, remove_columns=raw["train"].column_names)
    split  = dpo_ds.train_test_split(test_size=data_cfg.val_split, seed=data_cfg.seed)

    logger.info(
        "DPO split → train: %s | val: %s",
        f"{len(split['train']):,}",
        f"{len(split['test']):,}",
    )
    logger.info("Sample prompt: %s", split["train"][0]["prompt"][:120])
    return split["train"], split["test"]


# ── Training ───────────────────────────────────────────────────────────────────
def train_dpo(model, ref_model, tokenizer, train_ds, val_ds, model_cfg: DPOModelConfig) -> None:
    dpo_cfg = DPOConfig(
        output_dir=model_cfg.output_dir,
        beta=0.5,
        loss_type="simpo",
        learning_rate=5e-6,
        num_train_epochs=2,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        lr_scheduler_type="cosine",
        warmup_ratio=0.15,
        weight_decay=0.01,
        fp16=False,
        logging_steps=10,
        save_steps=50,
        save_total_limit=3,
        eval_steps=50,
        evaluation_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        report_to="none",
        gradient_checkpointing=True,
        optim="paged_adamw_8bit",
        max_grad_norm=0.3,
        max_length=1024,
        max_prompt_length=512,
        gamma=0.5,
        label_smoothing=0.1,
        push_to_hub=model_cfg.push_to_hub,
        hub_model_id=model_cfg.hub_model_id if model_cfg.push_to_hub else None,
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=dpo_cfg,
        train_dataset=train_ds,
        eval_dataset=val_ds,
    )

    logger.info("Starting DPO training with SimPO...")
    trainer.train()

    trainer.save_model(model_cfg.output_dir)
    tokenizer.save_pretrained(model_cfg.output_dir)
    logger.info("DPO adapter saved → %s", model_cfg.output_dir)

    # Push to HuggingFace Hub if configured
    if model_cfg.push_to_hub and model_cfg.hub_model_id:
        from huggingface_hub import HfApi
        api = HfApi()
        api.create_repo(repo_id=model_cfg.hub_model_id, repo_type="model", exist_ok=True)
        api.upload_folder(
            folder_path=model_cfg.output_dir,
            repo_id=model_cfg.hub_model_id,
            repo_type="model",
        )
        logger.info("Model pushed to HuggingFace → %s", model_cfg.hub_model_id)

    del trainer
    gc.collect()
    torch.cuda.empty_cache()


# ── Entrypoint ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        raise EnvironmentError("HF_TOKEN env variable not set. Run: export HF_TOKEN=hf_...")
    login(token=hf_token)

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    logger.info("Using CUDA device: %s", os.environ["CUDA_VISIBLE_DEVICES"])

    parser = HfArgumentParser((DPODataConfig, DPOModelConfig))
    data_cfg, model_cfg = parser.parse_args_into_dataclasses()

    set_seed(data_cfg.seed)

    # ── Tokenizer ──────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(model_cfg.model_id, use_fast=True)
    tokenizer.pad_token    = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # ── Data ───────────────────────────────────────────────────
    train_ds, val_ds = build_dpo_dataset(data_cfg)

    # ── Models ─────────────────────────────────────────────────
    merged  = load_and_merge_sft(model_cfg)
    model   = apply_dpo_lora(merged, model_cfg)
    ref_model = load_ref_model(model_cfg)

    # ── Train ──────────────────────────────────────────────────
    train_dpo(model, ref_model, tokenizer, train_ds, val_ds, model_cfg)

    del model, ref_model
    gc.collect()
    torch.cuda.empty_cache()
    logger.info("Done.")