"""One ledger entry per logical send; byte partitions are mutually exclusive."""
from __future__ import annotations

from collections import Counter
import math


class CommunicationCounter:
    def __init__(self, *, retain_events: bool = True) -> None:
        self.retain_events = retain_events
        self.events: list[dict] = []
        self._count = 0
        self._ledger: dict[tuple[str, str], dict] = {}

    def add_message(self, sender: int, receiver: int, phase: str,
                    message_type: str, payload_bytes: int = 0,
                    control_bytes: int = 0, **metadata) -> dict:
        if phase not in ("gt", "verification"):
            raise ValueError("Phase must be gt (protocol preparation) or verification")
        for name, value in (("payload_bytes", payload_bytes), ("control_bytes", control_bytes)):
            if isinstance(value, bool) or int(value) != value or value < 0:
                raise ValueError(f"{name} must be a nonnegative byte count")
        event = dict(metadata)
        event.update(message_id=self._count, sender=int(sender), receiver=int(receiver),
                     phase=phase, message_type=str(message_type),
                     payload_bytes=int(payload_bytes), control_bytes=int(control_bytes),
                     total_bytes=int(payload_bytes + control_bytes))
        for field in ('start_ms', 'end_ms', 'transfer_ms', 'network_ms'):
            if field in event and (not math.isfinite(event[field]) or event[field] < 0):
                raise AssertionError(f'Invalid message timing: {field}')
        if 'end_ms' in event and event['end_ms'] < event['start_ms']:
            raise AssertionError('Message arrived before it was sent')
        key = (phase, str(message_type))
        ledger = self._ledger.setdefault(key, dict(phase=phase, message_type=str(message_type),
            message_count=0, payload_bytes=0, control_bytes=0, total_bytes=0, control_messages=0))
        ledger['message_count'] += 1
        ledger['payload_bytes'] += int(payload_bytes)
        ledger['control_bytes'] += int(control_bytes)
        ledger['total_bytes'] += int(payload_bytes + control_bytes)
        ledger['control_messages'] += int(control_bytes > 0)
        self._count += 1
        if self.retain_events:
            self.events.append(event)
        return event

    def phase_totals(self) -> list[dict]:
        """Exact aggregate ledger; retaining individual events is optional."""
        return [dict(self._ledger[key]) for key in sorted(self._ledger)]

    def totals(self) -> dict:
        # Controls from BOTH phases are in control_bytes. The other two fields
        # contain payloads only, so no byte belongs to two summands.
        ledger = self.phase_totals()
        gt = sum(e["payload_bytes"] for e in ledger if e["phase"] == "gt")
        verification = sum(e["payload_bytes"] for e in ledger if e["phase"] == "verification")
        control = sum(e["control_bytes"] for e in ledger)
        return dict(gt_bytes=gt, verification_bytes=verification, control_bytes=control,
                    total_bytes=gt + verification + control, message_count=self._count,
                    gt_messages=sum(e['message_count'] for e in ledger if e['phase']=='gt'),
                    verification_messages=sum(e['message_count'] for e in ledger if e['phase']=='verification'),
                    control_messages=sum(e['control_messages'] for e in ledger),
                    gt_phase_total_bytes=sum(e["total_bytes"] for e in ledger if e["phase"] == "gt"))

    def validate(self) -> None:
        totals = self.totals()
        assert totals['total_bytes'] == sum(e['total_bytes'] for e in self.phase_totals())
        if self.retain_events:
            assert totals["total_bytes"] == sum(e["total_bytes"] for e in self.events)
            assert totals["message_count"] == len(self.events)
            assert [e["message_id"] for e in self.events] == list(range(len(self.events)))
        for e in self.events:
            for field in ("start_ms", "end_ms", "transfer_ms", "network_ms"):
                if field in e and (not math.isfinite(e[field]) or e[field] < 0):
                    raise AssertionError(f"Invalid message timing: {field}")
            if "end_ms" in e and e["end_ms"] < e["start_ms"]:
                raise AssertionError("Message arrived before it was sent")
