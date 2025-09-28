from gradio_i18n import Translate
from gradio_i18n import gettext as _


class Localizable:
    """A small wrapper for UI strings that defers calls to gettext until
    runtime.

    - `unwrap()` returns the original (English) text and is used where the
      codebase needs the raw value for storage/comparison.
    - `str(obj)` will attempt to translate the text via `_()`; if no gradio
      request context is available (which raises LookupError) it falls back
      to the original text. This avoids calling `_()` at module import time.
    """

    def __init__(self, text: str):
        self._text = text

    def __str__(self) -> str:
        try:
            return _(self._text)
        except LookupError:
            # No gradio request context available — return the base string.
            return self._text

    def unwrap(self) -> str:
        """Return the raw (untranslated) text used for storage/comparison."""
        return self._text

    def __eq__(self, other) -> bool:  # pragma: no cover - trivial
        if isinstance(other, Localizable):
            return self._text == other._text
        # compare against plain strings too
        return self._text == other

    from gradio_i18n import gettext as _

    class Localizable:
        """A small wrapper for UI strings that defers calls to gettext until
        runtime.

        - `unwrap()` returns the original (English) text and is used where the
          codebase needs the raw value for storage/comparison.
        - `str(obj)` will attempt to translate the text via `_()`; if no gradio
          request context is available (which raises LookupError) it falls back
          to the original text. This avoids calling `_()` at module import time.
        """

        def __init__(self, text: str):
            self._text = text

        def __str__(self) -> str:
            try:
                return _(self._text)
            except LookupError:
                # No gradio request context available — return the base string.
                return self._text

        def unwrap(self) -> str:
            """Return the raw (untranslated) text used for storage/comparison."""
            return self._text

        def __eq__(self, other) -> bool:  # pragma: no cover - trivial
            if isinstance(other, Localizable):
                return self._text == other._text
            # compare against plain strings too
            return self._text == other

        from gradio_i18n import gettext as _

        class Localizable:
            """A small wrapper for UI strings that defers calls to gettext until
            runtime.

            - `unwrap()` returns the original (English) text and is used where the
              codebase needs the raw value for storage/comparison.
            - `str(obj)` will attempt to translate the text via `_()`; if no gradio
              request context is available (which raises LookupError) it falls back
              to the original text. This avoids calling `_()` at module import time.
            """

            def __init__(self, text: str):
                self._text = text

            def __str__(self) -> str:
                try:
                    return _(self._text)
                except LookupError:
                    # No gradio request context available — return the base string.
                    return self._text

            def unwrap(self) -> str:
                """Return the raw (untranslated) text used for storage/comparison."""
                return self._text

            def __eq__(self, other) -> bool:  # pragma: no cover - trivial
                if isinstance(other, Localizable):
                    return self._text == other._text
                # compare against plain strings too
                return self._text == other

            def __repr__(self) -> str:
                return f"Localizable({self._text!r})"

        # Public constants used around the codebase.
        AUTOMATIC_DETECTION = Localizable("Automatic Detection")
        GRADIO_NONE_STR = ""
        GRADIO_NONE_NUMBER_MAX = 9999
        GRADIO_NONE_NUMBER_MIN = 0
        GRADIO_NONE_NUMBER_MIN = 0
