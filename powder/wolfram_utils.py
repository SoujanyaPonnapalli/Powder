#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import platform
from pathlib import Path
import wolframclient
import yaml
from wolframclient.evaluation import WolframLanguageSession
from wolframclient.language.expression import WLFunction
from wolframclient.language import wl, wlexpr, WLInputExpression
from typing import Optional
from fractions import Fraction

class WolframSessionManager:
    """
    Class that holds WolframSessions and will terminate them on deconstruction
    """
    def __init__(self, kernel_path: str):
        self.kernel_path = kernel_path
        self.sessions = []

    def start_session(self) -> WolframLanguageSession:
        """
        Start a new Wolfram Language session and return it
        """
        self.sessions.append(WolframLanguageSession(self.kernel_path))
        return self.sessions[-1]

    def cleanup_session(self, session: WolframLanguageSession) -> None:
        """
        Cleanup a session and remove it from the list of sessions
        """
        session.terminate()
        self.sessions.remove(session)

    def terminate_sessions(self):
        for session in self.sessions:
            session.terminate()
        self.sessions = []

    def __del__(self):
        print("One WolframSessionManager is terminating its Wolfram Language sessions")
        for session in self.sessions:
            session.terminate()


def _default_config_path() -> Path:
    """Find config.yaml at the project root (parent of powder/ package)."""
    return Path(__file__).resolve().parent.parent / "config.yaml"


def get_wolfram_kernel_path(config_path: Optional[str] = None) -> str:
    env_kernel_path = os.getenv("WOLFRAM_KERNEL_PATH")
    if env_kernel_path:
        return env_kernel_path

    resolved_path = Path(config_path) if config_path else _default_config_path()
    if not resolved_path.exists():
        raise FileNotFoundError(
            f"Missing Wolfram config at {resolved_path}. "
            "Set WOLFRAM_KERNEL_PATH or create config.yaml."
        )

    with resolved_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}

    kernel_paths = config.get("wolfram", {}).get("kernel_paths", {})
    system_name = platform.system().lower()
    if system_name.startswith("darwin"):
        os_key = "darwin"
    elif system_name.startswith("linux"):
        os_key = "linux"
    elif system_name.startswith("windows"):
        os_key = "windows"
    else:
        os_key = system_name

    kernel_path = kernel_paths.get(os_key) or kernel_paths.get("default")
    if not kernel_path:
        raise KeyError(
            f"No Wolfram kernel path configured for '{os_key}'. "
            "Update config.yaml or set WOLFRAM_KERNEL_PATH."
        )

    return kernel_path

def convert_wolfram_number(n) -> float:
    if type(n) == int:
        return float(n)
    if type(n) == wolframclient.language.expression.WLFunction:
        return float(Fraction(n[0], n[1]))
    print(f"unexpected type {type(n)}")
    return

def get_mtt_state(wolfram_mm: WLFunction, state_id: int, wolfram_session: WolframLanguageSession) -> float:
    """
    Get the mean time to a state from a Wolfram Markov Model
    wolfram_mm: the markov model with predefined starting state distribution
    state_id: the 1 indexed state corresponding to a row/col in the wolfram_mm transition matrix
    """
    dist = wl.Mean(wl.FirstPassageTimeDistribution(wolfram_mm, state_id))

    return convert_wolfram_number(wolfram_session.evaluate(dist))
