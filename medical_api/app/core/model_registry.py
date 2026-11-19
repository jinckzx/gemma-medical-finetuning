from __future__ import annotations

import json
import os
import threading
from typing import Any, Dict, Optional


class ModelRegistry:
    """Singleton-style store for all loaded models (everything loads lazily on first use)."""

    def __init__(self):
        self._hf_pipeline: Optional[Any] = None
        self._whisper: Optional[Any] = None
        self._keras: Optional[Any] = None
        self._skin_meta: Optional[Dict[str, Any]] = None
        self._skin_path: Optional[str] = None
        self._lock = threading.RLock()

    # ------------------------------------------------------------------ #
    # MedGemma text pipeline
    # ------------------------------------------------------------------ #
    @property
    def hf_pipeline(self):
        if self._hf_pipeline is None:
            with self._lock:
                if self._hf_pipeline is None:
                    self._hf_pipeline = self._build_text_pipeline()
        return self._hf_pipeline

    def _build_text_pipeline(self):
        import torch
        from peft import PeftModel
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
            pipeline,
        )
        from app.core.config import settings

        dtype = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }.get(settings.torch_dtype, torch.float16)
        token = settings.hf_token or None

        load_kwargs: Dict[str, Any] = dict(device_map=settings.device_map, token=token)
        use_4bit = (
            settings.load_in_4bit
            and torch.cuda.is_available()
            and settings.device_map != "cpu"
        )
        if use_4bit:
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=dtype,
            )

        # base model (newer transformers use `dtype`, older use `torch_dtype`)
        try:
            model = AutoModelForCausalLM.from_pretrained(
                settings.base_model_id, dtype=dtype, **load_kwargs
            )
        except TypeError:
            model = AutoModelForCausalLM.from_pretrained(
                settings.base_model_id, torch_dtype=dtype, **load_kwargs
            )

        # DPO mode: the DPO adapter sits on top of the merged SFT model
        if settings.sft_adapter_id:
            model = PeftModel.from_pretrained(
                model, settings.sft_adapter_id, token=token
            ).merge_and_unload()

        # the adapter being served (SFT-only mode, or the DPO adapter)
        model = PeftModel.from_pretrained(model, settings.hf_model_id, token=token)
        model.eval()

        try:
            tokenizer = AutoTokenizer.from_pretrained(settings.hf_model_id, token=token)
        except Exception:
            tokenizer = AutoTokenizer.from_pretrained(settings.base_model_id, token=token)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        return pipeline("text-generation", model=model, tokenizer=tokenizer)

    # ------------------------------------------------------------------ #
    # Keras skin classifier
    # ------------------------------------------------------------------ #
    def _resolve_skin_path(self) -> str:
        """Return the weights path, downloading from the Hub first if configured."""
        if self._skin_path:
            return self._skin_path

        from app.core.config import settings

        path = settings.keras_model_path
        if not os.path.exists(path):
            if not settings.skin_model_repo:
                raise FileNotFoundError(
                    f"Skin model weights not found at '{path}'. Train the model with "
                    "skin_disease_image_model/training.py, or set SKIN_MODEL_REPO to a "
                    "Hugging Face repo containing model.weights.h5, model_meta.json "
                    "and class_names.json."
                )
            from huggingface_hub import snapshot_download

            snapshot_download(
                repo_id=settings.skin_model_repo,
                local_dir=settings.skin_model_dir,
                token=settings.hf_token or None,
            )
            path = os.path.join(settings.skin_model_dir, os.path.basename(path))
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"'{os.path.basename(path)}' not found in repo {settings.skin_model_repo}"
                )
        self._skin_path = path
        return path

    @property
    def skin_meta(self) -> Dict[str, Any]:
        """model type, image_size, preprocessing and class_names written by the training script."""
        if self._skin_meta is None:
            with self._lock:
                if self._skin_meta is None:
                    folder = os.path.dirname(os.path.abspath(self._resolve_skin_path()))
                    meta_path = os.path.join(folder, "model_meta.json")
                    names_path = os.path.join(folder, "class_names.json")
                    for p in (meta_path, names_path):
                        if not os.path.exists(p):
                            raise FileNotFoundError(
                                f"{p} not found. It is written next to the weights by the "
                                "skin classifier training script."
                            )
                    with open(meta_path) as f:
                        meta = json.load(f)
                    with open(names_path) as f:
                        meta["class_names"] = json.load(f)
                    self._skin_meta = meta
        return self._skin_meta

    @property
    def keras(self):
        if self._keras is None:
            with self._lock:
                if self._keras is None:
                    path = self._resolve_skin_path()
                    if path.endswith(".weights.h5"):
                        self._keras = self._build_skin_model(path)
                    else:  # a full saved model
                        from tensorflow.keras.models import load_model

                        self._keras = load_model(path)
        return self._keras

    def _build_skin_model(self, weights_path: str):
        import tensorflow as tf
        from tensorflow.keras import layers
        from tensorflow.keras.applications import EfficientNetB3, ResNet50

        meta = self.skin_meta
        h, w = meta["image_size"]
        base_cls = EfficientNetB3 if meta["model"] == "efficientnet" else ResNet50
        base = base_cls(weights=None, include_top=False, input_shape=(h, w, 3))
        x = layers.GlobalAveragePooling2D()(base.output)
        x = layers.Dense(512, activation="relu")(x)
        x = layers.Dropout(0.5)(x)
        out = layers.Dense(meta["num_classes"], activation="softmax")(x)
        model = tf.keras.Model(base.input, out)
        model.load_weights(weights_path)
        return model

    def preprocess_skin_image(self, image):
        """PIL image -> (1, H, W, 3) float32 array, preprocessed exactly like in training."""
        import numpy as np

        meta = self.skin_meta
        h, w = meta["image_size"]
        arr = np.asarray(image.convert("RGB").resize((w, h)), dtype="float32")
        if meta["model"] != "efficientnet":
            from tensorflow.keras.applications.resnet50 import preprocess_input

            arr = preprocess_input(arr)
        # EfficientNet rescales internally, so it takes raw 0-255 pixels (no /255)
        return arr[None, ...]

    def predict_skin(self, image, top_k: int = 3) -> Dict[str, Any]:
        """Classify a PIL image. Returns the best label, its confidence and the top-k list."""
        import numpy as np

        probs = self.keras.predict(self.preprocess_skin_image(image), verbose=0)[0]
        names = self.skin_meta["class_names"]
        order = np.argsort(probs)[::-1][:top_k]
        top = [{"label": names[i], "confidence": float(probs[i])} for i in order]
        return {"label": top[0]["label"], "confidence": top[0]["confidence"], "top_k": top}

    # ------------------------------------------------------------------ #
    def reset(self):
        with self._lock:
            self._hf_pipeline = None
            self._whisper = None
            self._keras = None
            self._skin_meta = None
            self._skin_path = None


registry = ModelRegistry()