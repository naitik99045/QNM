"""
Literal Symbol Guard Module.
Protects exact string literals (file paths, URLs, code blocks, flags, identifiers)
from lossy VQ-VAE quantization and restores them losslessly.
"""

import re
import json
from typing import Dict, Tuple, List, Optional
from pydantic import BaseModel, Field


class SymbolTable(BaseModel):
    """Bidirectional lookup table for masked symbols."""
    symbol_to_literal: Dict[str, str] = Field(default_factory=dict)
    literal_to_symbol: Dict[str, str] = Field(default_factory=dict)
    counter: int = 0

    def register(self, literal: str) -> str:
        """Register a literal and return its assigned $SYM_n placeholder."""
        if literal in self.literal_to_symbol:
            return self.literal_to_symbol[literal]

        symbol = f"$SYM_{self.counter}"
        self.counter += 1
        self.symbol_to_literal[symbol] = literal
        self.literal_to_symbol[literal] = symbol
        return symbol

    def restore(self, text: str) -> str:
        """Replace all $SYM_n placeholders with their original literals."""
        # Sort symbols in descending order of length to avoid prefix collisions (e.g. $SYM_10 before $SYM_1)
        sorted_symbols = sorted(self.symbol_to_literal.keys(), key=lambda s: len(s), reverse=True)
        for sym in sorted_symbols:
            text = text.replace(sym, self.symbol_to_literal[sym])
        return text

    def to_json(self) -> str:
        return json.dumps({
            "symbol_to_literal": self.symbol_to_literal,
            "literal_to_symbol": self.literal_to_symbol,
            "counter": self.counter
        })

    @classmethod
    def from_json(cls, data: str) -> "SymbolTable":
        d = json.loads(data)
        return cls(**d)


class SymbolGuard:
    """Scanner that detects exact string literals, masks them, and restores them."""

    # Regex patterns ordered from highest specificity (code blocks, urls) to lower (flags, identifiers)
    PATTERNS: List[Tuple[str, re.Pattern]] = [
        ("CODE_BLOCK", re.compile(r"```[a-zA-Z0-9_\-]*\n[\s\S]*?```")),
        ("INLINE_CODE", re.compile(r"`[^`\n]+`")),
        ("URL", re.compile(r"https?://[^\s()<>\[\]'\"]+")),
        ("WIN_PATH", re.compile(r"[a-zA-Z]:\\(?:[^\\/:*?\"<>|\s\r\n]+\\)*[^\\/:*?\"<>|\s\r\n]+")),
        ("POSIX_PATH", re.compile(r"(?:\.{1,2}/|[a-zA-Z0-9_\.\-]+/)+[a-zA-Z0-9_\.\-]+\.[a-zA-Z0-9]+")),
        ("CLI_FLAG", re.compile(r"(?:--[a-zA-Z0-9_\-]+(?:=[^\s]+)?|-[a-zA-Z0-9]+)")),
        ("ENV_VAR", re.compile(r"(?:\$[A-Z_][A-Z0-9_]{1,}|%[A-Z_][A-Z0-9_]{1,}%)")),
        ("CAMEL_IDENTIFIER", re.compile(r"\b[a-z]+(?:[A-Z][a-z0-9]+)+\b")),
        ("SNAKE_IDENTIFIER", re.compile(r"\b[a-z][a-z0-9]*(?:_[a-z0-9]+)+\b")),
    ]

    def __init__(self):
        pass

    def mask(self, text: str, existing_table: Optional[SymbolTable] = None) -> Tuple[str, SymbolTable]:
        """
        Scans input text, replaces exact literals with $SYM_n placeholders,
        and returns the masked text along with the updated SymbolTable.
        """
        table = existing_table or SymbolTable()
        masked_text = text

        for pattern_name, pattern in self.PATTERNS:
            def replacer(match: re.Match) -> str:
                matched_literal = match.group(0)
                # Don't re-mask already created symbols
                if matched_literal.startswith("$SYM_"):
                    return matched_literal
                sym = table.register(matched_literal)
                return sym

            masked_text = pattern.sub(replacer, masked_text)

        return masked_text, table

    def restore(self, text: str, table: SymbolTable) -> str:
        """Restores all symbols in the text using the provided SymbolTable."""
        return table.restore(text)
