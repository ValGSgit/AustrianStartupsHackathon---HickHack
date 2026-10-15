"""Turn one instruction into MolmoAct2 task sentences, one per object.

The dashboard's router writes plain English ("Put the duck and the chess queen into the cup"); MolmoAct2 was
trained on one object per sentence in a fixed wording. So:

    split_tasks("put the duck and the chess queen into the cup")
    -> ["move the arm towards the duck, grasp it, lift it up, and drop it into the cup",
        "move the arm towards the chess queen, grasp it, lift it up, and drop it into the cup"]

Sentences already in the model's wording are kept. Anything this can't read (no container named) is passed on
unchanged, as one task.
"""

import re

MODEL_PREFIX = "move the arm towards the "
VERBS = r"(?:please\s+)?(?:can you\s+)?(?:pick up|pick|put|place|drop|move|grab|take|get|bring|throw|toss|lift)\s+(?:up\s+)?"
# greedy head: the LAST "in/into/on ..." names the container (up to 3 words, e.g. "white cup")
CONTAINER = r"^(.*)\s+(into|inside|onto|on top of|in|on|to)\s+(?:the\s+|a\s+|an\s+)?([a-z\-]+(?: [a-z\-]+){0,2})$"
FILLER = {"", "it", "them", "both", "all", "each", "everything"}


def model_sentence(obj: str, prep: str, container: str) -> str:
    how = "put it on the" if prep in ("on", "onto", "on top of") else "drop it into the"
    return f"{MODEL_PREFIX}{obj}, grasp it, lift it up, and {how} {container}"


def _objects(text: str) -> list[str]:
    text = re.sub(r"^" + VERBS, "", text.strip())
    parts = re.split(r"\s*,\s*(?:and\s+)?|\s+and\s+|\s+then\s+|\s*&\s*|\s+as well as\s+", text)
    out = []
    for p in parts:
        p = re.sub(r"^(?:both|all|also)\s+", "", p.strip())
        p = re.sub(r"^(?:the|a|an)\s+", "", p)
        p = re.sub(r"^" + VERBS, "", p)
        p = re.sub(r"^(?:the|a|an)\s+", "", p).strip()
        if p not in FILLER:
            out.append(p)
    return out


def _clause(text: str) -> list[str]:
    if text.startswith(MODEL_PREFIX):
        return [text]
    m = re.match(CONTAINER, text)
    if not m:
        return [text] if text else []
    head, prep, container = m.group(1), m.group(2), m.group(3).strip()
    # "the duck into the cup and the queen into the cup": the head itself names a container -> split it first
    inner = re.search(r"^(.*?)\s+(?:into|in|inside|onto|on)\s+(?:the\s+)?[a-z\- ]+?,?\s+(?:and|then)\s+(.*)$", head)
    if inner:
        return _clause(inner.group(1) + f" {prep} the {container}") + _clause(inner.group(2) + f" {prep} the {container}")
    return [model_sentence(o, prep, container) for o in _objects(head)] or [text]


def split_tasks(instruction: str) -> list[str]:
    text = instruction.strip().lower().rstrip(".!? ")
    text = re.sub(r"[\"'“”‘’]", "", text)
    text = re.sub(r"(?:\s*,?\s+(?:too|as well|please|both|for me))+$", "", text)
    clauses = [c.strip() for c in re.split(r"\s*;\s*|\.\s+|\s*,?\s+and then\s+|\s*,\s*then\s+", text) if c.strip()]
    # "the duck into the cup, then the queen": a clause without a container gets the one named elsewhere
    named = next((m for c in clauses if not c.startswith(MODEL_PREFIX) and (m := re.match(CONTAINER, c))), None)
    tasks = []
    for c in clauses:
        if named and not c.startswith(MODEL_PREFIX) and not re.match(CONTAINER, c):
            c = f"{c} {named.group(2)} the {named.group(3)}"
        tasks += _clause(c)
    return tasks


if __name__ == "__main__":
    import sys

    for t in split_tasks(" ".join(sys.argv[1:])):
        print(t)
