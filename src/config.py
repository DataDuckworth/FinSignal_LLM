"""One place for secrets and private settings (API keys, ...). Nothing in here is ever committed.

How it works
- Put the values in a file called `.env` in the project root, one KEY=value per line (see `.env.example`).
  `.env` is listed in .gitignore, so it stays on your machine.
- load_env() copies them into os.environ. A variable that is already set in the real environment wins, so a
  one-off `$env:GROQ_API_KEY = ...` in the shell still overrides the file.
- Code asks for a value with get_secret("NAME"); it never has to know where it came from.

Why not a Python dictionary in a .py file: a .py file is exactly the kind of file that gets committed or pasted by
accident. A .env file is the usual convention, and the same file works for any tool that reads .env.

Deliberately tiny: this does not replace python-dotenv's full syntax (no multi-line values, no variable expansion).
"""
import os
import pathlib

ENV_FILE = pathlib.Path(__file__).resolve().parents[1] / ".env"      # project root, whatever the current folder is


def _parse(text):
    """KEY=value lines -> dict. Skips blank lines and # comments; accepts 'export KEY=value' and quoted values."""
    values = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].lstrip()
        if "=" not in line:
            continue
        name, value = line.split("=", 1)
        name, value = name.strip(), value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]                                         # strip matching quotes
        if name and value:                                              # an empty value means "not filled in yet"
            values[name] = value
    return values


def load_env(path=ENV_FILE):
    """Load the .env file into os.environ without overriding variables that are already set. Returns the names loaded."""
    path = pathlib.Path(path)
    if not path.exists():
        return []
    # utf-8-sig: Windows editors such as Notepad often add an invisible byte-order mark to the first line
    values = _parse(path.read_text(encoding="utf-8-sig"))
    loaded = []
    for name, value in values.items():
        if not os.environ.get(name):
            os.environ[name] = value
            loaded.append(name)
    return loaded


def get_secret(name):
    """Return a secret or raise a clear error. The error never contains the value."""
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"{name} is not set. Put it in {ENV_FILE} (copy .env.example) or set it in the shell.")
    return value


def secret_status(*names):
    """Safe-to-print summary such as {'GROQ_API_KEY': 'set (56 characters)'}; the values are never shown."""
    return {n: (f"set ({len(os.environ[n])} characters)" if os.environ.get(n) else "NOT set") for n in names}
