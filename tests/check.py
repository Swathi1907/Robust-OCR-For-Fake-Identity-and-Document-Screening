import base64
from pathlib import Path
import sys
import time
import requests


def encode_image_to_base64(image_path: str) -> str:
    path = Path(image_path)
    if not path.is_file():
        raise FileNotFoundError(f"Image not found at path: {image_path}")

    with open(path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode("utf-8")


def test_predict(
    image_path: str,
    doc_id: int = 3,
    request_id: int = 101,
    url: str = "http://127.0.0.1:8000/predict",
):
    print(f"Encoding '{image_path}' to base64...")
    b64_string = encode_image_to_base64(image_path)

    payload = {"image_base64": b64_string}

    headers = {
        "doc-id": str(doc_id),
        "request-id": str(request_id),
        "Content-Type": "application/json",
    }

    print(
        f"Sending request to {url} (doc_id={doc_id}, request_id={request_id})..."
    )

    start_time = time.perf_counter()
    response = requests.post(url, json=payload, headers=headers)
    elapsed_time = (time.perf_counter() - start_time) * 1000

    print(f"HTTP Status Code: {response.status_code}")
    print(f"Total Client Round-Trip: {elapsed_time:.2f} ms")
    print(f"Server Processing Time: {response.elapsed.total_seconds() * 1000:.2f} ms")

    try:
        print("Response JSON:")
        print(response.json())
    except Exception:
        print("Raw Response Text:")
        print(response.text)


if __name__ == "__main__":
    target_image = sys.argv[1] if len(sys.argv) > 1 else "MRZ.jpg"
    test_predict(target_image, doc_id=3, request_id=1)