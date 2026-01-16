"""
Shared model cache for HuggingFace models.

This module provides a global cache for HF models to be shared
between components that need read-only access (like User simulator).
"""

from typing import Optional, Tuple
from loguru import logger


class HFModelCache:
    """Global cache for HuggingFace models."""

    _models: dict = {}
    _tokenizers: dict = {}

    @classmethod
    def get_or_load_model(
        cls,
        model_name_or_path: str,
        device: str = "auto",
        torch_dtype: str = "auto",
        trust_remote_code: bool = True,
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
        shared: bool = True,  # If False, always load a new instance
    ) -> Tuple:
        """
        Get model from cache or load it.

        Args:
            model_name_or_path: Model path
            device: Device to use
            torch_dtype: Torch dtype
            trust_remote_code: Trust remote code
            load_in_8bit: Load in 8-bit
            load_in_4bit: Load in 4-bit
            shared: If True, use cached model. If False, load new instance.

        Returns:
            tuple: (model, tokenizer)
        """
        cache_key = model_name_or_path

        # If shared and already cached, return cached version
        if shared and cache_key in cls._models:
            logger.info(f"[HFModelCache] Using cached model: {model_name_or_path}")
            return cls._models[cache_key], cls._tokenizers[cache_key]

        # Load new model instance
        if shared:
            logger.info(f"[HFModelCache] Loading model (will be cached): {model_name_or_path}")
        else:
            logger.info(f"[HFModelCache] Loading new model instance (not cached): {model_name_or_path}")

        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError:
            raise ImportError(
                "Please install transformers and torch: "
                "pip install transformers torch"
            )

        # Determine torch dtype
        if torch_dtype == "auto":
            dtype = "auto"
        elif torch_dtype == "float16":
            dtype = torch.float16
        elif torch_dtype == "bfloat16":
            dtype = torch.bfloat16
        elif torch_dtype == "float32":
            dtype = torch.float32
        else:
            dtype = "auto"

        # Load tokenizer (always shared, no parameters)
        if cache_key in cls._tokenizers:
            tokenizer = cls._tokenizers[cache_key]
        else:
            tokenizer = AutoTokenizer.from_pretrained(
                model_name_or_path,
                trust_remote_code=trust_remote_code,
            )
            cls._tokenizers[cache_key] = tokenizer

        # Prepare model loading kwargs
        model_kwargs = {
            "trust_remote_code": trust_remote_code,
            "device_map": device,
        }

        if dtype != "auto":
            model_kwargs["torch_dtype"] = dtype

        if load_in_8bit:
            model_kwargs["load_in_8bit"] = True
        elif load_in_4bit:
            model_kwargs["load_in_4bit"] = True

        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            **model_kwargs,
        )

        # Cache the model only if shared
        if shared:
            cls._models[cache_key] = model
            logger.info(f"[HFModelCache] Model loaded and cached: {model_name_or_path}")
        else:
            logger.info(f"[HFModelCache] Model loaded (not cached): {model_name_or_path}")

        return model, tokenizer

    @classmethod
    def clear_cache(cls):
        """Clear the model cache to free memory."""
        cls._models.clear()
        cls._tokenizers.clear()
        import gc
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
        logger.info("[HFModelCache] Cache cleared")
