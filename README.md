# python_ollama
A Python script that helps you talk to a local Ollama install.

- Main module: `ollama_client.py` (`OllamaClient`, `ChatSession`)
- Requires the `ollama` Python package (`pip install ollama`, preferably in a virtual environment)

## CLI

```bash
python ollama_client.py list                         # installed models
python ollama_client.py ask "What is DNS?"           # single question
cat auth.log | python ollama_client.py ask "Summarise suspicious activity" -s "You are a security analyst"
python ollama_client.py chat                         # interactive, keeps history (/reset, /exit)
python ollama_client.py embed "some text"            # embedding vector as JSON
```

Global options (before the command): `--host URL` (env `OLLAMA_HOST`) and `-m/--model NAME`
(env `OLLAMA_MODEL`; default is the first installed model).

## Python

```python
from ollama_client import OllamaClient, ChatSession

client = OllamaClient()
session = ChatSession(client, model="llama3", system="Be brief.")
print(session.send("Hi, I'm Andreas."))
print(session.send("What's my name?"))   # remembers the previous turn
```

## Tests

```bash
python -m unittest
```

Uses only the standard library. The tests run against a small fake Ollama server
(`tests/fake_ollama.py`), so no real Ollama or model is needed.
