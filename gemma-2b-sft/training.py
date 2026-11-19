"""
training.py: QLoRA SFT fine-tuning for MedGemma-2B (google/gemma-2b)

Works with both older and newer versions of transformers / trl / datasets
(unsupported TrainingArguments / SFTConfig keys are dropped automatically).

Quick test:
    python training.py --save_dir /content/medical_dataset --num_train_epochs 1 \
        --max_train_samples 2000 --output_dir /content/model_test --push_to_hub False

Real run (reuse cached dataset):
    python training.py --save_dir /content/medical_dataset --load_from_cache True \
        --lora_r 32 --lora_alpha 64 --max_seq_length 1024 --num_train_epochs 1 \
        --max_train_samples 30000 --output_dir /content/model \
        --push_to_hub True --hub_model_id anmolsehgal/medgemma-2b-sft-v2
"""

import gc
import inspect
import logging
import os
import random
import sys
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import torch
from datasets import (
    DatasetDict,
    concatenate_datasets,
    load_dataset,
    load_from_disk,
)
from huggingface_hub import get_token, login
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    set_seed,
)
from trl import SFTConfig, SFTTrainer

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


# ── Config dataclasses ─────────────────────────────────────────────────────────
@dataclass
class DataConfig:
    save_dir: str = field(default="medical_dataset", metadata={"help": "Directory to save/load processed dataset"})
    load_from_cache: bool = field(default=False, metadata={"help": "Skip dataset download and load from save_dir"})
    train_ratio: float = field(default=0.80)
    val_ratio: float = field(default=0.10)
    test_ratio: float = field(default=0.10)
    min_text_length: int = field(default=80, metadata={"help": "Min chars to keep a sample"})
    dedup_prefix_len: int = field(default=120, metadata={"help": "Chars used for near-dedup key"})
    seed: int = field(default=42)
    use_medqa: bool = field(default=True, metadata={"help": "Include MedQA dataset"})
    use_mmlu: bool = field(default=True, metadata={"help": "Include MMLU medical subsets"})


@dataclass
class ModelConfig:
    model_id: str = field(default="google/gemma-2b")
    output_dir: str = field(default="model")
    lora_r: int = field(default=32, metadata={"help": "LoRA rank"})
    lora_alpha: int = field(default=64, metadata={"help": "LoRA alpha"})
    lora_dropout: float = field(default=0.1, metadata={"help": "LoRA dropout"})
    target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        metadata={"help": "Target modules for LoRA"},
    )
    use_rslora: bool = field(default=True, metadata={"help": "Use rank-scaled LoRA"})
    max_seq_length: int = field(default=1024, metadata={"help": "Max sequence length"})
    num_train_epochs: int = field(default=1)
    max_steps: int = field(default=-1, metadata={"help": "Overrides epochs if > 0"})
    max_train_samples: int = field(default=0, metadata={"help": "0 = use all training samples"})
    max_eval_samples: int = field(default=500, metadata={"help": "Cap on validation samples"})
    per_device_train_batch_size: int = field(default=2)
    gradient_accumulation_steps: int = field(default=8)
    learning_rate: float = field(default=5e-5)
    eval_steps: int = field(default=200)
    packing: bool = field(default=True)
    push_to_hub: bool = field(default=False)
    hub_model_id: Optional[str] = field(default=None)


# ── Reproducibility ────────────────────────────────────────────────────────────
def set_reproducibility(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    set_seed(seed)
    torch.backends.cudnn.deterministic = True


# ── Dataset helpers ────────────────────────────────────────────────────────────
def _safe_load(names, *args, **kwargs):
    """Try several dataset ids in order and return the first that loads."""
    last_err = None
    for name in names:
        try:
            return load_dataset(name, *args, **kwargs)
        except Exception as e:  # noqa: BLE001
            last_err = e
            logger.warning("load_dataset(%s) failed: %s", name, str(e)[:200])
    raise last_err


def _to_text_parts(raw, fmt) -> list:
    return [ds.map(fmt, remove_columns=ds.column_names) for ds in raw.values()]


def _load_medqa() -> list:
    """MedQA (USMLE, 4 options)."""
    raw = _safe_load(["GBaker/MedQA-USMLE-4-options"])

    def fmt(x):
        question = (x.get("question") or "").strip()
        options = x.get("options") or {}
        answer = (x.get("answer") or "").strip()
        letter = (x.get("answer_idx") or "").strip()
        opts_str = "\n".join(f"{k}) {v}" for k, v in options.items() if v)
        ans = f"{letter}) {answer}" if letter else answer
        return {"text": f"Question:\n{question}\n\nOptions:\n{opts_str}\n\nAnswer: {ans}".strip()}

    return _to_text_parts(raw, fmt)


MMLU_MEDICAL_SUBSETS = [
    "anatomy", "clinical_knowledge", "college_medicine", "medical_genetics",
    "professional_medicine", "nutrition", "virology",
]


def _load_mmlu_medical() -> list:
    parts = []
    for subset in MMLU_MEDICAL_SUBSETS:
        try:
            raw = _safe_load(["cais/mmlu"], subset)

            def fmt(x, subset=subset):
                question = (x.get("question") or "").strip()
                choices = list(x.get("choices") or [])
                idx = x.get("answer", 0)
                answer = choices[idx] if choices and 0 <= idx < len(choices) else ""
                opts_str = "\n".join(f"{chr(65 + i)}) {c}" for i, c in enumerate(choices))
                return {"text": f"[{subset}]\nQuestion: {question}\nOptions:\n{opts_str}\nAnswer: {answer}".strip()}

            new = _to_text_parts(raw, fmt)
            parts.extend(new)
            logger.info("MMLU-%s loaded: %s samples", subset, sum(len(s) for s in new))
        except Exception as e:  # noqa: BLE001
            logger.warning("MMLU-%s skipped: %s", subset, str(e)[:200])
    return parts


def _load_pubmedqa() -> list:
    parts = []
    for config_name in ("pqa_labeled", "pqa_unlabeled"):
        try:
            raw = _safe_load(["qiaojin/PubMedQA", "pubmed_qa"], config_name)

            def fmt(x):
                ctx = " ".join(x["context"]["contexts"])[:1000]
                ans = (x.get("long_answer") or "").strip()
                return {"text": f"{x['question'].strip()}\n{ctx}\n{ans}".strip()}

            parts.extend(_to_text_parts(raw, fmt))
        except Exception as e:  # noqa: BLE001
            logger.warning("PubMedQA config=%s skipped: %s", config_name, str(e)[:200])
    return parts


def _load_medmcqa() -> list:
    raw = _safe_load(["openlifescienceai/medmcqa", "medmcqa"])

    def fmt(x):
        opts = (
            f"A) {x.get('opa', '')}\n"
            f"B) {x.get('opb', '')}\n"
            f"C) {x.get('opc', '')}\n"
            f"D) {x.get('opd', '')}"
        )
        exp = (x.get("exp") or "").strip()
        subject = (x.get("subject_name") or "").strip()
        topic = (x.get("topic_name") or "").strip()
        header = f"[{subject} - {topic}]\n" if subject else ""
        cop = x.get("cop", -1)
        ans = f"\nAnswer: {'ABCD'[cop]}" if isinstance(cop, int) and 0 <= cop <= 3 else ""
        return {"text": f"{header}{x['question'].strip()}\n{opts}{ans}\n{exp}".strip()}

    return _to_text_parts(raw, fmt)


def _load_medalpaca() -> list:
    raw = _safe_load(["medalpaca/medical_meadow_medical_flashcards"])

    def fmt(x):
        fields = [x.get("instruction", ""), x.get("input", ""), x.get("output", "")]
        return {"text": "\n".join(f.strip() for f in fields if f and f.strip())}

    return _to_text_parts(raw, fmt)


def build_dataset(cfg: DataConfig) -> DatasetDict:
    """Download, clean, dedup, and split the medical corpus."""
    if cfg.load_from_cache:
        logger.info("Loading dataset from cache: %s", cfg.save_dir)
        return load_from_disk(cfg.save_dir)

    assert abs(cfg.train_ratio + cfg.val_ratio + cfg.test_ratio - 1.0) < 1e-6

    loaders = {
        "PubMedQA": _load_pubmedqa,
        "MedMCQA": _load_medmcqa,
        "MedAlpaca": _load_medalpaca,
    }
    if cfg.use_medqa:
        loaders["MedQA"] = _load_medqa
    if cfg.use_mmlu:
        loaders["MMLU-Medical"] = _load_mmlu_medical

    parts = []
    for name, loader_fn in loaders.items():
        try:
            splits = loader_fn()
            n = sum(len(s) for s in splits)
            parts.extend(splits)
            logger.info("%-12s loaded: %s samples", name, f"{n:,}")
        except Exception as e:  # noqa: BLE001
            logger.error("%-12s FAILED: %s", name, str(e)[:300])

    if not parts:
        raise RuntimeError("All dataset loaders failed.")

    combined = concatenate_datasets(parts)
    before = len(combined)

    combined = combined.filter(
        lambda x: isinstance(x["text"], str) and len(x["text"].strip()) >= cfg.min_text_length
    )
    logger.info("After length filter: %s -> %s", f"{before:,}", f"{len(combined):,}")

    seen: set = set()

    def is_unique(x):
        key = x["text"][: cfg.dedup_prefix_len].strip().lower()
        if key in seen:
            return False
        seen.add(key)
        return True

    combined = combined.filter(is_unique)
    logger.info("After dedup: %s samples", f"{len(combined):,}")

    combined = combined.shuffle(seed=cfg.seed)
    total = len(combined)
    n_train = int(total * cfg.train_ratio)
    n_val = int(total * cfg.val_ratio)

    ds_dict = DatasetDict({
        "train": combined.select(range(n_train)),
        "val": combined.select(range(n_train, n_train + n_val)),
        "test": combined.select(range(n_train + n_val, total)),
    })

    logger.info(
        "Split -> train: %s | val: %s | test: %s",
        f"{len(ds_dict['train']):,}",
        f"{len(ds_dict['val']):,}",
        f"{len(ds_dict['test']):,}",
    )

    ds_dict.save_to_disk(cfg.save_dir)
    logger.info("Dataset cached to: %s", cfg.save_dir)
    return ds_dict


# ── Model helpers ──────────────────────────────────────────────────────────────
def _use_bf16() -> bool:
    return torch.cuda.is_available() and torch.cuda.is_bf16_supported()


def load_model_and_tokenizer(cfg: ModelConfig):
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_id, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    compute_dtype = torch.bfloat16 if _use_bf16() else torch.float16
    logger.info("Compute dtype: %s", compute_dtype)

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
    )

    load_kwargs = dict(quantization_config=bnb, device_map="auto")
    try:
        model = AutoModelForCausalLM.from_pretrained(cfg.model_id, dtype=compute_dtype, **load_kwargs)
    except (TypeError, ValueError):
        model = AutoModelForCausalLM.from_pretrained(cfg.model_id, torch_dtype=compute_dtype, **load_kwargs)

    model = prepare_model_for_kbit_training(model)
    model.config.use_cache = False

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

    model = get_peft_model(model, LoraConfig(**lora_kwargs))
    model.print_trainable_parameters()
    return model, tokenizer


def _filter_kwargs(cls, kwargs: dict) -> dict:
    """Drop keys the installed version of `cls` does not accept."""
    params = inspect.signature(cls.__init__).parameters
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()) and cls is not SFTConfig:
        return kwargs
    ok = {k: v for k, v in kwargs.items() if k in params}
    dropped = sorted(set(kwargs) - set(ok))
    if dropped:
        logger.warning("%s: ignoring unsupported args: %s", cls.__name__, dropped)
    return ok


# ── Training ───────────────────────────────────────────────────────────────────
def train(model, tokenizer, model_cfg: ModelConfig, train_ds, val_ds) -> None:
    adapter_path = os.path.join(model_cfg.output_dir, "lora_adapter")
    bf16 = _use_bf16()
    sft_params = inspect.signature(SFTConfig.__init__).parameters

    args = dict(
        output_dir=model_cfg.output_dir,
        num_train_epochs=model_cfg.num_train_epochs,
        max_steps=model_cfg.max_steps,
        per_device_train_batch_size=model_cfg.per_device_train_batch_size,
        per_device_eval_batch_size=model_cfg.per_device_train_batch_size,
        gradient_accumulation_steps=model_cfg.gradient_accumulation_steps,
        learning_rate=model_cfg.learning_rate,
        lr_scheduler_type="cosine",
        weight_decay=0.02,
        bf16=bf16,
        fp16=not bf16,
        logging_steps=10,
        save_steps=model_cfg.eval_steps,
        save_total_limit=2,
        eval_steps=model_cfg.eval_steps,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        report_to="none",
        dataloader_num_workers=2,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="paged_adamw_8bit",
        max_grad_norm=0.5,
        push_to_hub=False,  # adapter is uploaded manually at the end
        dataset_text_field="text",
        packing=model_cfg.packing,
    )
    # renamed across versions
    args["eval_strategy" if "eval_strategy" in sft_params else "evaluation_strategy"] = "steps"
    args["max_length" if "max_length" in sft_params else "max_seq_length"] = model_cfg.max_seq_length
    if "warmup_ratio" in sft_params:
        args["warmup_ratio"] = 0.1
    else:
        args["warmup_steps"] = 0.1

    training_args = SFTConfig(**_filter_kwargs(SFTConfig, args))

    trainer_kwargs = dict(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
    )
    trainer_params = inspect.signature(SFTTrainer.__init__).parameters
    trainer_kwargs["processing_class" if "processing_class" in trainer_params else "tokenizer"] = tokenizer
    trainer = SFTTrainer(**trainer_kwargs)

    logger.info("Starting training...")
    trainer.train()

    os.makedirs(adapter_path, exist_ok=True)
    trainer.save_model(adapter_path)
    tokenizer.save_pretrained(adapter_path)
    logger.info("LoRA adapter saved -> %s", adapter_path)

    if model_cfg.push_to_hub and model_cfg.hub_model_id:
        from huggingface_hub import HfApi

        api = HfApi()
        api.create_repo(repo_id=model_cfg.hub_model_id, repo_type="model", exist_ok=True)
        api.upload_folder(
            folder_path=adapter_path,
            repo_id=model_cfg.hub_model_id,
            repo_type="model",
        )
        logger.info("Model pushed to HuggingFace -> %s", model_cfg.hub_model_id)

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

    parser = HfArgumentParser((DataConfig, ModelConfig))
    data_cfg, model_cfg = parser.parse_args_into_dataclasses()

    set_reproducibility(data_cfg.seed)

    # Data
    ds = build_dataset(data_cfg)
    train_ds, val_ds = ds["train"], ds["val"]
    if model_cfg.max_train_samples:
        train_ds = train_ds.select(range(min(model_cfg.max_train_samples, len(train_ds))))
    val_ds = val_ds.select(range(min(model_cfg.max_eval_samples, len(val_ds))))
    logger.info("Training on %s samples, evaluating on %s", f"{len(train_ds):,}", f"{len(val_ds):,}")

    # Model
    model, tokenizer = load_model_and_tokenizer(model_cfg)

    # Train
    train(model, tokenizer, model_cfg, train_ds, val_ds)

    logger.info("Done.")