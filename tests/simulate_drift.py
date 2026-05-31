import asyncio
import websockets
import json
import uuid
import time
import subprocess
import requests

API_URL = "ws://localhost:8001/ws"
ALICE_ID = "531c7f56-e5c4-4557-9b2d-e7e8ed7c942f"

async def test_anomaly():
    print("=== Starting Drift & Anomaly Simulation ===")
    
    # 0. Login to get cookie
    print("[0] Logging in...")
    login_resp = requests.post("http://localhost:8001/auth/demo-login/0")
    if login_resp.status_code != 200:
        print("Login failed:", login_resp.text)
        return
    cookie = login_resp.cookies.get("_req_sid")
    
    print("[1] Injecting highly entropic garbage message to trigger model divergence...")

    session_id = str(uuid.uuid4())
    url = f"{API_URL}/{ALICE_ID}?session_id={session_id}"
    headers = {"Cookie": f"_req_sid={cookie}"}

    garbage_text = "aslkdjfasdjf klsdjflkasdjf asdjflkasdjflk asdjflkasdjflkasdjflkasdjf" * 10
    
    async with websockets.connect(url, additional_headers=headers) as ws:
        msg_id = str(uuid.uuid4())
        payload = {
            "text": garbage_text,
            "conversation_id": "conv-drift-1",
            "message_id": msg_id,
            "timestamp": time.time()
        }
        await ws.send(json.dumps(payload))
        print(f"Sent message: {msg_id}")

        # Wait for the response
        try:
            while True:
                resp = await asyncio.wait_for(ws.recv(), timeout=10.0)
                data = json.loads(resp)
                if data.get("type") == "message_ack":
                    continue
                if data.get("type") == "analysis":
                    print("\n[2] Received Aggregation Result:")
                    print(json.dumps(data.get("data", {}), indent=2))
                    break
        except asyncio.TimeoutError:
            print("❌ Timed out waiting for analysis.")
            return

    print("\n[3] Checking Docker logs for Anomaly Detection Fallback...")
    # Tail the central_responder_service logs to prove Dynamic Recovery kicked in
    time.sleep(2)
    result = subprocess.run(
        ["docker", "logs", "--tail", "50", "finalproject-main-central_responder_service-1"],
        capture_output=True, text=True
    )
    logs = result.stdout + result.stderr

    if "Anomaly Fallback: Dynamic Recovery" in logs and msg_id in logs:
        print("\n✅ SUCCESS: Dynamic Recovery detected the anomaly and handled it gracefully!")
    else:
        print("\n❌ FAILED: Did not find Dynamic Recovery log for this message.")

if __name__ == "__main__":
    asyncio.run(test_anomaly())
