#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import wolframclient
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
