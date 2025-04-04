#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import wolframclient
from fractions import Fraction

def convert_wolfram_number(n) -> float:
    if type(n) == int:
        return float(n)
    if type(n) == wolframclient.language.expression.WLFunction:
        return float(Fraction(n[0], n[1]))
    print(f"unexpected type {type(n)}")
    return
