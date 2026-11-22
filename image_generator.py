import requests
import os
from dotenv import load_dotenv
from main import WorkFlowState
from pathlib import Path

load_dotenv()

def image_process(image):

    account_id = os.getenv("CLOUDFLARE_ACCOUNT_ID")
    api_key = os.getenv("CLOUDFLARE_API_TOKEN")

    url = (
        f"https://api.cloudflare.com/client/v4/accounts/"
        f"{account_id}/ai/run/@cf/black-forest-labs/flux-1-schnell"
    )

    headers = {
        "Authorization": f"Bearer {api_key}"
    }

    payload = {
        "prompt": image.prompt
    }

    try:
        response = requests.post(
            url,
            headers=headers,
            json=payload,
            timeout=120
        )

        response.raise_for_status()

        data = response.json()

        import base64

        image_data = base64.b64decode(data["result"]["image"])

        output_dir = Path(__file__).resolve().parent / "Generated_blogs" / "images"
        output_dir.mkdir(parents=True, exist_ok=True)

        output_path = output_dir / Path(image.filename).name

        with open(output_path, "wb") as file:
            file.write(image_data)
        print("Image generated successfully!")

    except requests.exceptions.RequestException as e:
        print(f"API request failed: {e}")

    except (KeyError, ValueError) as e:
        print(f"Unexpected image response: {e}")

    return {
        "message" : "successful image generation"
    }