from typing import List

import attr


@attr.s()
class TextChunk(object):
    text: str = attr.ib()
    summary: str = attr.ib()
    # Whitespace that follows the text.
    ws: str = attr.ib(default="")


@attr.s()
class TextFull(object):
    texts: List[TextChunk] = attr.ib()

    def __str__(self) -> str:
        result = list()
        for i, ti in enumerate(self.texts):
            result.append(ti.text)
            if i == len(self.texts) - 1:
                result.append(ti.ws)
        return "".join(result)

    def __repr__(self) -> str:
        return str(self)

    def summary(self) -> str:
        return " ".join([t.summary for t in self.texts])


def wrap_text_in_chunks(text: str, summary: str) -> TextFull:
    return TextFull(texts=[TextChunk(text=text, summary=summary)])


def wrap_summary(summary_func):
    def wrapped_func(text, model) -> TextFull:
        return wrap_text_in_chunks(text, summary_func(text, model))
    return wrapped_func
