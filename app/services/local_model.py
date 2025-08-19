# app/services/local_model.py
from pathlib import Path
from typing import Iterator, List, Dict, Literal, Optional

# 선택 1) HuggingFace Transformers
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
    import torch
    _HAS_HF = True
except Exception:
    _HAS_HF = False

# 선택 2) llama.cpp (GGUF)
try:
    from llama_cpp import Llama
    _HAS_LLAMA_CPP = True
except Exception:
    _HAS_LLAMA_CPP = False

class LocalModel:
    def __init__(
        self,
        backend: Literal["hf", "llama_cpp"],
        model_path: str | Path,
        device: Optional[str] = None,
        max_new_tokens: int = 512,
        temperature: float = 0.2,
    ):
        self.backend = backend
        self.model_path = str(model_path)
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        if backend == "hf":
            if not _HAS_HF:
                raise RuntimeError("transformers/torch 미설치. requirements.txt 확인")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                use_fast=True,
                trust_remote_code=True,
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None,
                trust_remote_code=True,
            ).to(self.device)
        elif backend == "llama_cpp":
            if not _HAS_LLAMA_CPP:
                raise RuntimeError("llama-cpp-python 미설치. requirements.txt 확인")
            # model_path에는 .gguf 파일 경로를 지정
            self.model = Llama(
                model_path=self.model_path,
                n_gpu_layers=-1 if self._has_cuda() else 0,
                n_ctx=4096,
                logits_all=False,
                verbose=False,
            )
        else:
            raise ValueError("backend는 'hf' 또는 'llama_cpp'만 지원")

    def _has_cuda(self) -> bool:
        try:
            import torch
            return torch.cuda.is_available()
        except Exception:
            return False

    def build_prompt(self, messages: List[Dict[str, str]]) -> str:
        """
        messages: [{"role":"system"|"user"|"assistant", "content": "..."}]
        심플 시스템 프롬프트 + 대화 템플릿
        """
        sys = "당신은 한국어로 간결하고 정확하게 답하는 유능한 도우미입니다."
        parts = [f"<|system|>\n{sys}\n"]
        for m in messages:
            role = m["role"]
            if role == "user":
                parts.append(f"<|user|>\n{m['content']}\n")
            elif role == "assistant":
                parts.append(f"<|assistant|>\n{m['content']}\n")
        parts.append("<|assistant|>\n")  # 모델이 답변을 이어서 생성
        return "".join(parts)

    def stream(self, messages: List[Dict[str, str]]) -> Iterator[str]:
        """토큰 단위 스트리밍"""
        if self.backend == "hf":
            prompt = self.build_prompt(messages)
            streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
            inputs = self.tokenizer([prompt], return_tensors="pt").to(self.device)
            gen_kwargs = dict(
                **inputs,
                streamer=streamer,
                max_new_tokens=self.max_new_tokens,
                do_sample=True if self.temperature > 0 else False,
                temperature=self.temperature,
                top_p=0.9,
                repetition_penalty=1.05,
                eos_token_id=self.tokenizer.eos_token_id,
            )
            # 비동기 스레드로 생성 시작
            import threading
            thread = threading.Thread(target=self.model.generate, kwargs=gen_kwargs)
            thread.start()
            for token in streamer:
                yield token
            thread.join()
        else:  # llama_cpp
            prompt = self.build_prompt(messages)
            for out in self.model.create_completion(
                prompt=prompt,
                max_tokens=self.max_new_tokens,
                temperature=self.temperature,
                top_p=0.9,
                stream=True,
                stop=["<|user|>", "<|system|>", "<|assistant|>"],
            ):
                yield out["choices"][0]["text"]

try:
    from langchain_community.llms import LlamaCpp
except Exception as _e:
    LlamaCpp = None

def load_llama(
    model_path: str,
    *,
    n_ctx: int = 4096,
    n_batch: int = 512,
    n_threads: int | None = None,
    n_gpu_layers: int = 0,          # GPU VRAM 허용 범위에서 크게(예: 999)
    temperature: float = 0.1,
    max_tokens: int = 256,
    stop: list[str] | None = None,
):
    """
    LangChain 호환 LlamaCpp LLM을 로드하여 반환.
    main.py의 build_chain()에서 그대로 사용 가능합니다.
    """
    if LlamaCpp is None:
        raise RuntimeError("langchain-community가 필요합니다. requirements.txt 확인")

    import os
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"GGUF model not found: {model_path}")

    # 대화 템플릿 맞춤형 stop 토큰
    stop = stop or ["<|user|>", "<|system|>", "<|assistant|>", "</s>"]

    llm = LlamaCpp(
        model_path=model_path,
        n_ctx=n_ctx,
        n_batch=n_batch,
        n_threads=n_threads or (os.cpu_count() or 4),
        n_gpu_layers=n_gpu_layers,
        temperature=temperature,
        max_tokens=max_tokens,
        stop=stop,
        streaming=True,   # LCEL과 함께 사용 시 astream 가능
        verbose=False,
    )
    return llm
