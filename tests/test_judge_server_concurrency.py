# Copyright 2021 ETH Zurich and the OptArena authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""The judge HTTP server (service.py) is a ThreadingHTTPServer, so grade requests arrive
concurrently. It must sequentialize the TIMED grade per device -- at most one timed grade per
DeviceSlot -- or concurrent timings contend and the speedup ratio is corrupted. This test fires
more concurrent /score requests than there are slots and asserts the server never runs more than
`len(slots)` grades at once. No real compile / GPU / LLM: `score` is faked to probe concurrency."""
import dataclasses
import json
import threading
import time
import urllib.request

from optarena.agent_bench import service
from optarena.agent_bench.judge_scheduler import DeviceSlot
from optarena.api import InputMode


@dataclasses.dataclass(frozen=True)
class FakeResult:
    build_ok: bool = True
    correct: bool = True
    speedup: float = 1.0


class ConcurrencyProbe:
    """Stands in for score(): records the peak number of grades running at once."""

    def __init__(self):
        self.active = 0
        self.peak = 0
        self.lock = threading.Lock()

    def __call__(self, *args, **kwargs):
        with self.lock:
            self.active += 1
            self.peak = max(self.peak, self.active)
        try:
            time.sleep(0.05)  # hold the "device" long enough for overlap to show if unbounded
            return FakeResult()
        finally:
            with self.lock:
                self.active -= 1


def test_judge_server_bounds_concurrent_grades_to_device_slots(monkeypatch):
    probe = ConcurrencyProbe()
    monkeypatch.setattr(service, "score", probe)
    real_get = service.config.get  # capture BEFORE patching (else the lambda recurses)
    monkeypatch.setattr(service.config,
                        "get",
                        lambda key, default=None: False if key == "record.enabled" else real_get(key, default))
    slots = [DeviceSlot("cpu", 0, None), DeviceSlot("cpu", 1, None)]  # exactly 2 timing slots
    cfg = dataclasses.replace(service.from_config(), input_mode=InputMode.EITHER)
    server = service.make_server("127.0.0.1", 0, cfg, slots=slots)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    def fire():
        body = json.dumps({"kernel": "gemm", "language": "c", "source": "int x;"}).encode()
        req = urllib.request.Request(f"http://127.0.0.1:{port}/score",
                                     data=body,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            assert resp.status == 200
            json.loads(resp.read())

    try:
        firers = [threading.Thread(target=fire) for _ in range(6)]
        for t in firers:
            t.start()
        for t in firers:
            t.join()
    finally:
        server.shutdown()

    assert probe.peak >= 1  # grades actually ran
    assert probe.peak <= 2  # never more than the 2 device slots at once
