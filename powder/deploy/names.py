"""Random ``AdjectiveNoun`` handles for deployments (plan §4.1), e.g. ``OrangePanda``.

A handle names one deployment. It is unique across live AWS tags and existing result directories (the caller
passes both sets as ``taken``), is never reused, and is also accepted explicitly via ``--handle``.
"""

from __future__ import annotations

import random
import re
from typing import Iterable

ADJECTIVES = """
amber ancient arctic autumn bold brave breezy bright brisk calm careful cheerful clever cobalt cosmic crimson
crisp curious dapper daring deep distant dusty eager electric emerald fancy fearless fluffy frosty gentle
giant gilded glad golden graceful grand happy hazel honest humble icy indigo jade jolly keen kind lively lucky
lunar mellow merry mighty misty modest nimble noble orange patient peaceful plucky polite proud purple quick
quiet radiant rapid rosy royal rustic sandy scarlet serene shiny silent silver sleepy smooth snowy solar
sparkling speedy spry steady stormy sunny swift tidy tiny tranquil trusty velvet vivid warm wild windy wise
witty young zesty
""".split()

NOUNS = """
albatross antelope badger beaver bison bobcat buffalo camel cardinal caribou cheetah chipmunk condor cougar
coyote crane cricket dolphin dragonfly eagle egret elk falcon ferret finch firefly flamingo fox gazelle gecko
giraffe goose gopher hamster hare hawk hedgehog heron hummingbird ibis jackal jaguar kestrel kingfisher koala
lark lemur leopard lion llama lynx magpie manatee marmot meerkat mongoose moose narwhal newt ocelot osprey
otter owl oyster panda panther parrot pelican penguin pheasant puffin quail rabbit raccoon raven robin salmon
seal sparrow squirrel stork swan tiger toucan turtle walrus weasel whale wolf wombat yak zebra
""".split()

assert len(set(ADJECTIVES)) == len(ADJECTIVES) and len(set(NOUNS)) == len(NOUNS)

_VALID = re.compile(r"^[A-Za-z][A-Za-z0-9-]{2,30}$")


class HandleError(ValueError):
    pass


def validate(handle: str) -> str:
    """Return ``handle`` if it is acceptable as an explicit name, else raise HandleError."""
    if handle.lower() == "auto":
        raise HandleError("'auto' is the placeholder for a generated handle, not a valid name")
    if not _VALID.match(handle):
        raise HandleError(
            f"invalid handle {handle!r}: 3-31 chars, start with a letter, then letters, digits or '-'"
        )
    return handle


def generate(taken: Iterable[str] = (), rng: random.Random | None = None, max_tries: int = 1000) -> str:
    """A random ``AdjectiveNoun`` not in ``taken`` (compared case-insensitively)."""
    rng = rng or random.SystemRandom()
    used = {t.lower() for t in taken}
    if len(used) >= len(ADJECTIVES) * len(NOUNS):
        raise HandleError("handle space exhausted")
    for _ in range(max_tries):
        h = rng.choice(ADJECTIVES).capitalize() + rng.choice(NOUNS).capitalize()
        if h.lower() not in used:
            return h
    raise HandleError(f"could not find an unused handle in {max_tries} tries")


def ensure_new(handle: str, taken: Iterable[str]) -> str:
    """Validate an explicit handle and require that it is not already in use."""
    validate(handle)
    if handle.lower() in {t.lower() for t in taken}:
        raise HandleError(f"handle {handle!r} is already in use (live deployment or results directory)")
    return handle
