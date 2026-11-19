from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    All values can be overridden with environment variables (case-insensitive)
    or a .env file, e.g.  HF_MODEL_ID=anmolsehgal/medgemma-2b-dpo-v2
    """

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # ── Text model (MedGemma) ──────────────────────────────────────────────
    base_model_id: str = "google/gemma-2b"
    # LoRA adapter to serve. Use the SFT adapter, or the DPO adapter (see sft_adapter_id).
    # Can be a Hugging Face repo id or a local folder such as /content/model/lora_adapter.
    hf_model_id: str = "anmolsehgal/medgemma-2b-sft"
    # Only set this when hf_model_id is the DPO adapter. The DPO adapter was trained on top of
    # the merged SFT model, so the SFT adapter must be merged first.
    sft_adapter_id: str = ""
    hf_token: str = ""                  # needed for the gated google/gemma-2b download
    torch_dtype: str = "float16"        # "bfloat16" (A100/L4) | "float16" (T4) | "float32"
    device_map: str = "auto"            # "auto" | "cpu" | "cuda"
    load_in_4bit: bool = True           # matches how the adapters were trained; needs a GPU
    max_new_tokens: int = 300

    # ── Skin classifier (Keras) ────────────────────────────────────────────
    # model.weights.h5 must sit next to model_meta.json and class_names.json
    # (all three are written by skin_disease_image_model/training.py).
    keras_model_path: str = "./model.weights.h5"
    # Optional: Hugging Face repo holding those three files. If keras_model_path does not
    # exist, the repo is downloaded into skin_model_dir on first use.
    skin_model_repo: str = ""
    skin_model_dir: str = "./skin_model"
    confidence_threshold: float = 0.5


settings = Settings()