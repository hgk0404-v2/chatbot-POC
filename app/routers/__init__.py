# app/routers/__init__.py
"""
app/routers 폴더 안의 모든 모듈을 자동으로 import 하여,
각 모듈이 노출한 `router`(fastapi.APIRouter)를 FastAPI 앱에 등록합니다.

사용법:
    from app.routers import include_all_routers
    include_all_routers(app)
"""

from importlib import import_module
from pkgutil import iter_modules
from pathlib import Path
from fastapi import FastAPI, APIRouter
from typing import Iterable

PKG_NAME = __name__                      # "app.routers"
PKG_PATH = Path(__file__).parent         # .../app/routers


def _iter_router_modules() -> Iterable[str]:
    """app/routers/ 하위의 .py 모듈들(패키지는 제외)을 찾아 모듈 경로 문자열로 yield"""
    for info in iter_modules([str(PKG_PATH)]):
        if info.ispkg:
            continue
        # __init__.py 자신이나, 언더스코어로 시작하는 내부 모듈은 스킵(원하면 규칙 변경 가능)
        if info.name.startswith("_") or info.name == "__init__":
            continue
        yield f"{PKG_NAME}.{info.name}"


def include_all_routers(app: FastAPI) -> None:
    """
        각 모듈에서 노출한 객체들을 찾아 등록 규칙:
        - 우선순위1: 모듈 전역에 `routers: list[APIRouter]` 가 있으면 전부 include
        - 우선순위2: 모듈 전역에 `router: APIRouter` 가 있으면 단일 include
        - (선택) 모듈 전역에 `prefix: str` 이 있으면 prefix 적용
        - (선택) 모듈 전역에 `tags: list[str]` 이 있으면 tags 적용
    """
    for mod_path in _iter_router_modules():
        mod = import_module(mod_path)

        # 설정값(optional)
        prefix = getattr(mod, "prefix", "")
        tags = getattr(mod, "tags", None)

        # 1) routers 리스트가 있으면 우선
        routers = getattr(mod, "routers", None)
        if routers and isinstance(routers, (list, tuple)):
            for r in routers:
                if isinstance(r, APIRouter):
                    app.include_router(r, prefix=prefix, tags=tags)
            continue

        # 2) router 단일 객체가 있으면
        r = getattr(mod, "router", None)
        if isinstance(r, APIRouter):
            app.include_router(r, prefix=prefix, tags=tags)
