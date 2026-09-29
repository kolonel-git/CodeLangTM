"""CodeLangTM: interpretable source code language identification via Tsetlin Machines."""

__version__ = "0.1.0"

CORE_LANGUAGES = ("python", "cpp", "java", "javascript", "rust", "go", "sql", "html")
STRETCH_LANGUAGES = ("c", "csharp", "typescript", "kotlin", "php", "ruby")
ALL_LANGUAGES = CORE_LANGUAGES + STRETCH_LANGUAGES

# The languages the evaluation code (baselines, TM, reports) uses by default. Still the 8 core
# ones until the Stage B dataset (v6) exists; collection, build and audit already know all 14.
LANGUAGES = CORE_LANGUAGES
