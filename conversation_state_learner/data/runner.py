"""Pipeline runner — sends generated conversations through the live emotion pipeline"""

import asyncio
import json
import time
import logging
import requests
from typing import Dict, List, Optional

logger = logging.getLogger("runner")

PIPELINE_WAIT_PER_MSG = 1.0
MSG_SEND_INTERVAL = 0.5
PIPELINE_POLL_RETRIES = 10
PIPELINE_POLL_DELAY = 1.0


class PipelineRunner:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip("/")
        self._alice: Optional[Dict] = None
        self._bob:   Optional[Dict] = None

    # ── Auth ──────────────────────────────────────────────────────────────────

    def _login_demo(self, slot: int) -> dict:
        """Login as a demo user and return {user_id, display_name, cookie}."""
        session = requests.Session()
        resp = session.post(f"{self.base_url}/auth/demo-login/{slot}", timeout=10)
        resp.raise_for_status()
        data = resp.json()
        cookie = session.cookies.get("_req_sid")
        if not cookie:
            raise RuntimeError(f"No auth cookie returned for demo slot {slot}")
        return {
            "user_id":      data["user_id"],
            "display_name": data["display_name"],
            "cookie":       cookie,
        }

    def _ensure_users(self):
        if self._alice is None:
            logger.info("Logging in as Alice (slot 0)...")
            self._alice = self._login_demo(0)
            logger.info(f"  Alice: {self._alice['user_id']}")
        if self._bob is None:
            logger.info("Logging in as Bob (slot 1)...")
            self._bob = self._login_demo(1)
            logger.info(f"  Bob:   {self._bob['user_id']}")

    # ── Conversation creation ─────────────────────────────────────────────────

    def _create_conversation(self) -> str:
        """Create a direct conversation between Alice and Bob. Returns conv_id."""
        resp = requests.post(
            f"{self.base_url}/conversations",
            json={
                "user_id":        self._alice["user_id"],
                "target_user_id": self._bob["user_id"],
            },
            timeout=10,
        )
        resp.raise_for_status()
        conv_id = resp.json()["conversation_id"]
        logger.debug(f"  Conversation created: {conv_id}")
        return conv_id

    # ── Pipeline fetch ────────────────────────────────────────────────────────

    def _fetch_pipeline(self, message_id: str) -> Optional[Dict]:
        """Poll the admin pipeline endpoint until data is available."""
        headers = {"Cookie": f"_req_sid={self._alice['cookie']}"}
        for attempt in range(PIPELINE_POLL_RETRIES):
            try:
                resp = requests.get(
                    f"{self.base_url}/admin/pipeline/{message_id}",
                    headers=headers,
                    timeout=10,
                )
                if resp.status_code == 200:
                    data = resp.json()
                    if data.get("stages", {}).get("vader") or data.get("decision", {}).get("dominant"):
                        return data
                    logger.debug(f"  Pipeline data not ready yet for {message_id[:8]}, retrying...")
                elif resp.status_code == 404:
                    logger.debug(f"  Message {message_id[:8]} not in DB yet, retrying...")
                else:
                    logger.warning(f"  Pipeline fetch returned {resp.status_code} for {message_id[:8]}")
            except Exception as e:
                logger.warning(f"  Pipeline fetch error: {e}")
            time.sleep(PIPELINE_POLL_DELAY)
        logger.warning(f"  Pipeline data never arrived for {message_id[:8]}")
        return None

    # ── WebSocket send + capture ──────────────────────────────────────────────

    async def _ws_send_and_capture(
        self, conv_id: str, messages: List[str]
    ) -> List[Dict]:
        """Open a WebSocket as Alice, send all messages, collect analysis events."""
        try:
            import websockets
        except ImportError:
            raise ImportError("websockets package required: pip install websockets")

        ws_url = self.base_url.replace("http://", "ws://").replace("https://", "wss://")
        ws_url = f"{ws_url}/ws/{self._alice['user_id']}"
        headers = {"Cookie": f"_req_sid={self._alice['cookie']}"}

        analysis_events: List[Dict] = []
        total_messages = len(messages)

        tail_wait = PIPELINE_WAIT_PER_MSG * total_messages

        async with websockets.connect(ws_url, additional_headers=headers) as ws:
            logger.debug(f"  WebSocket connected to {ws_url}")

            async def sender():
                for i, text in enumerate(messages):
                    turn_sender = self._alice["user_id"] if i % 2 == 0 else self._bob["user_id"]
                    payload = json.dumps({
                        "text":            text,
                        "conversation_id": conv_id,
                        "sender_id":       turn_sender,
                    })
                    await ws.send(payload)
                    logger.debug(f"  Sent [{i+1}/{total_messages}]: {text[:60]}")
                    if i < total_messages - 1:
                        await asyncio.sleep(MSG_SEND_INTERVAL)
                await asyncio.sleep(tail_wait)
                await ws.close()

            async def receiver():
                try:
                    async for raw in ws:
                        try:
                            event = json.loads(raw)
                            if event.get("type") == "analysis":
                                analysis_events.append(event)
                                logger.debug(
                                    f"  Analysis event received: "
                                    f"{event.get('data', {}).get('final_dominant_emotion', '?')} "
                                    f"(id={str(event.get('data', {}).get('id', ''))[:8]})"
                                )
                        except json.JSONDecodeError:
                            pass
                except Exception:
                    pass

            await asyncio.gather(sender(), receiver())

        logger.debug(f"  Captured {len(analysis_events)} analysis events")
        return analysis_events

    # ── Main public method ────────────────────────────────────────────────────

    def run_conversation(
        self,
        trajectory_type: str,
        messages: List[str],
    ) -> dict:
        """Send a list of messages through the pipeline and return an enriched record."""
        self._ensure_users()
        conv_id = self._create_conversation()

        logger.info(f"  Running {len(messages)} messages through pipeline...")

        analysis_events = asyncio.run(
            self._ws_send_and_capture(conv_id, messages)
        )

        enriched_messages = []
        for i, text in enumerate(messages):
            event = analysis_events[i] if i < len(analysis_events) else None
            msg_id = (event or {}).get("data", {}).get("id")

            pipeline_data = None
            if msg_id:
                time.sleep(1.0)
                pipeline_data = self._fetch_pipeline(msg_id)

            enriched_messages.append({
                "turn":       i,
                "text":       text,
                "message_id": msg_id,
                "ws_event":   event,
                "pipeline":   pipeline_data,
            })

        success_count = sum(1 for m in enriched_messages if m["pipeline"] is not None)
        logger.info(
            f"  Done: {success_count}/{len(messages)} messages have full pipeline data"
        )

        return {
            "trajectory_type": trajectory_type,
            "conversation_id": conv_id,
            "collected_at":    time.time(),
            "messages":        enriched_messages,
        }
