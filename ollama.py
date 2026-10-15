import ollama
import requests
from typing import List, Optional, Dict, Any, Literal
import json

class OllamaClient:
    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url.rstrip("/")

    def check_connection(self) -> bool:
        """Tester om Ollama er tilgjengelig."""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except requests.exceptions.RequestException as e:
            print(f"Ollama-tilkoblingsfeil: {e}")
            return False

    def list_models(self) -> List[Dict[str, str]]:
        """Hent liste over tilgjengelige modeller."""
        tags_response = ollama.list()
        models = tags_response.get('models', [])
        return [
            {"name": model.get('name'), "modified_at": model.get('modified_at')}
            for model in models
        ]

    def generate(
        self,
        prompt: str,
        model: str,
        stream: bool = True,
        temperature: float = 0.7,
        max_tokens: int = None
    ):
        """Genererer en tekstrespons for gitt prompt."""
        try:
            response = ollama.chat(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                stream=stream
            )

            if not stream:
                return json.loads(response.read()).get('message', {'content': ''})

            return {
                'response': response,
                'model_used': model,
                'temperature': temperature
            }
        except requests.exceptions.ConnectionError:
            print(f"Feil: Kan ikke koble til Ollama på {self.base_url}")
            raise

    def batch_process(
        self,
        prompts: List[str],
        model: str
    ) -> Dict[int, Any]:
        """Behandle flere prompts samtidet ved hjelp av multithreading."""
        import concurrent.futures
        results = {}

        def process_prompt(index, prompt):
            try:
                result = ollama.chat(
                    model=model,
                    messages=[{"role": "user", "content": prompt}]
                )
                return (index, {"status": "success", "response": result})
            except Exception as e:
                return (index, {"status": "error", "error": str(e)})

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            future_to_index = {
                executor.submit(process_prompt, i, p): i for i, p in enumerate(prompts)
            }

            for future in concurrent.futures.as_completed(future_to_index):
                index, result = future.result()
                results[index] = result

        return results

    def system_message(
        self,
        prompt: str,
        model: Optional[str] = None
    ) -> Dict[str, Any]:
        """Tilføy en systemprompt eller instruksjonsbeskrivelse."""
        response = ollama.chat(
            model=model or "llama2",
            messages=[{"role": "system", "content": prompt}]
        )
        return response.get('message', {})

    def embed(
        self,
        prompt: str,
        model: Optional[str] = None
    ):
        """Generer en embeddingsvektorer for prompten."""
        response = ollama.embeddings(
            model=model or "nomic-embed-text",
            prompt=prompt
        )
        return response.get('embedding', [])

    def list_embeddings_models(self) -> List[str]:
        """Liste over modeller som støtter embeddings."""
        try:
            models = ollama.list()
            embed_models = [m['name'] for m in models.get('models', [])
                           if 'nomic' in m['name'].lower() or 'embed' in m['name'].lower()]
            return embed_models
        except Exception as e:
            print(f"Feil ved fetching embedding models: {e}")
            return []

    def run_agent(self, instructions: str, prompt: str):
        """Kjør en enkel agent-prompt med systeminstruksjoner."""
        messages = [
            {"role": "system", "content": f"{instructions}"}
        ]

        try:
            response = ollama.chat(
                model="llama2",  # Eller annen modell du bruker
                messages=messages,
                stream=True
            )

            output = []
            for chunk in response:
                content = chunk.get('message', {}).get('content')
                if content:
                    output.append(content)

            return "".join(output)
        except Exception as e:
            print(f"Agent kjøring feilet: {e}")
            return None

    def stream_generation(
        self,
        model: str,
        prompt: str
    ):
        """Streamet respons fra modellen."""
        try:
            response = ollama.chat(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                stream=True
            )

            for line in response:
                content = line.get('message', {}).get('content')
                if content:
                    print(f">>> {content}")
        except Exception as e:
            print(f"Streaming feilet: {e}")

# Brukeksjempel og testkoder
def main():
    # Initialisering av klienten
    client = OllamaClient()

    if not client.check_connection():
        print("Ollama er ikke tilgjengelig. Sørg for at den kjører.")
        return

    # List over modeller
    models = client.list_models()
    print("\nTilgjengelige modeller:")
    for model in models:
        print(f" - {model['name']}")

    prompt = "Si meg noe om kunstig intelligens på norsk."

    # Test generering
    result = client.generate(prompt, "llama2")
    if 'response' in dir(client):
        print(result)

    # Systemprompt eksempel
    system_response = client.system_message("Du er en norsk AI-assistent.")
    print(f"\nSystemprompt: {system_response}")

if __name__ == "__main__":
    main()
