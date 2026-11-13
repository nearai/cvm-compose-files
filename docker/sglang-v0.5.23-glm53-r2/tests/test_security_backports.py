"""CPU checks for media-local-path-reject, session-params-validation and caller-mm-hashes-opt-in.

Run inside the image with pytest (see test-cpu.sh); it imports the installed, patched package.
The other backports bring their own upstream unit tests, which test-cpu.sh runs.
"""

import ast
import io
import struct
import wave
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest

import sglang.srt.utils.common as common

SRT = Path(common.__file__).resolve().parents[1]
PNG_1X1 = bytes.fromhex(
    "89504e470d0a1a0a0000000d4948445200000001000000010806000000"
    "1f15c4890000000d49444154789c6360000002000154a24f5d0000000049454e44ae426082"
)


def _wav_bytes() -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(struct.pack("<160h", *([0] * 160)))
    return buf.getvalue()


@pytest.fixture
def media_files(tmp_path, monkeypatch):
    monkeypatch.delenv(common.LOCAL_MEDIA_ENV, raising=False)
    png = tmp_path / "secret.png"
    png.write_bytes(PNG_1X1)
    wav = tmp_path / "secret.wav"
    wav.write_bytes(_wav_bytes())
    mp4 = tmp_path / "secret.mp4"
    mp4.write_bytes(b"\x00\x00\x00\x18ftypmp42")
    return SimpleNamespace(png=str(png), wav=str(wav), mp4=str(mp4))


# --- media-local-path-reject (sglang#45206) ---------------------------------------------------


@pytest.mark.parametrize("scheme", ["", "file://"])
def test_image_local_paths_rejected(media_files, scheme):
    ref = scheme + media_files.png
    with pytest.raises(ValueError, match="Local file paths"):
        common.get_image_bytes(media_files.png)
    with pytest.raises(ValueError, match="Local file paths"):
        common.load_image(ref)


@pytest.mark.parametrize("scheme", ["", "file://"])
def test_audio_local_paths_rejected(media_files, scheme):
    with pytest.raises(ValueError, match="Local file paths"):
        common.load_audio(scheme + media_files.wav)


def test_video_local_paths_rejected(media_files):
    with pytest.raises(ValueError, match="Local file paths"):
        common._normalize_video_input("file://" + media_files.mp4)
    # A bare path is not probed: it is decoded as base64 and fails like any bad input.
    for fn in (common._normalize_video_input, common.get_video_bytes):
        with pytest.raises(ValueError):
            fn(media_files.mp4)


def test_local_paths_are_not_probed_without_opt_in(media_files, monkeypatch):
    from sglang.srt.multimodal.cache import identity

    def probe(*args, **kwargs):
        raise AssertionError("local path probed")

    monkeypatch.setattr(common.os.path, "isfile", probe)
    monkeypatch.setattr(identity.Path, "is_file", probe)
    for missing in (False, True):
        mp4 = media_files.mp4 + (".missing" if missing else "")
        png = "secret.png" if missing else media_files.png
        with pytest.raises(ValueError):
            common._normalize_video_input(mp4)
        with pytest.raises(ValueError):
            identity._read_media_bytes(png)


def test_mm_cache_identity_does_not_read_local_files(media_files):
    from sglang.srt.multimodal.cache import identity

    for ref in (media_files.png, "file://" + media_files.png):
        with pytest.raises(ValueError, match="Local file paths"):
            identity._read_media_bytes(ref)


def test_rejection_is_a_client_media_error(media_files):
    # ValueError is what the processors map to a 400 for bad client media.
    assert issubclass(ValueError, common.CLIENT_MEDIA_EXCEPTIONS)


def test_inline_media_still_accepted(media_files):
    import pybase64

    b64 = pybase64.b64encode(PNG_1X1).decode()
    assert common.get_image_bytes("data:image/png;base64," + b64) == PNG_1X1
    assert common.get_image_bytes(b64) == PNG_1X1
    assert common.get_image_bytes(PNG_1X1) == PNG_1X1


def test_local_paths_allowed_with_opt_in(media_files, monkeypatch):
    monkeypatch.setenv(common.LOCAL_MEDIA_ENV, "1")
    assert common.get_image_bytes(media_files.png) == PNG_1X1
    assert common._normalize_video_input("file://" + media_files.mp4) == media_files.mp4
    assert common._normalize_video_input(media_files.mp4) == media_files.mp4


# --- session-params-validation (CVE-2026-110809, CVE-2026-110807) ------------------------------


def _session_with_turns():
    from sglang.srt.session.session_controller import Session, SessionReqNode

    session = Session(capacity_of_str_len=1 << 16)
    reqs = {}
    parent = None
    for rid in ("t1", "t2", "t3"):
        req = SimpleNamespace(rid=rid, finished_reason=None, to_finish=None)
        node = SessionReqNode(req, parent)
        session.req_nodes[rid] = node
        reqs[rid] = req
        parent = node
    sibling = SimpleNamespace(rid="t2b", finished_reason=None, to_finish=None)
    session.req_nodes["t2b"] = SessionReqNode(sibling, session.req_nodes["t1"])
    reqs["t2b"] = sibling
    return session, reqs


def test_session_replace_without_rid_clears_every_turn():
    from sglang.srt.managers.io_struct import SessionParams

    session, reqs = _session_with_turns()
    made = {}

    def factory(**kwargs):
        new = mock.MagicMock(rid=kwargs["rid"])
        made["req"] = new
        return new

    req = mock.MagicMock()
    req.rid = "t4"
    req.input_ids = [1, 2, 3]
    req.session_params = SessionParams(id=session.session_id, replace=True)
    new_req = session.create_req(req, tokenizer=None, vocab_size=32, req_factory=factory)

    assert new_req is made["req"]
    new_req.set_finish_with_abort.assert_not_called()
    # Every earlier turn is aborted and gone; only the new turn is left.
    assert list(session.req_nodes) == ["t4"]
    assert all(r.to_finish is not None for r in reqs.values())


@pytest.mark.parametrize(
    "fields",
    [{"id": ["a"]}, {"id": {"x": 1}}, {"id": "s", "rid": 3}, {"id": "s", "offset": "abc"},
     {"id": "s", "replace": "yes"}, {"id": "s", "drop_previous_output": 1}],
)
def test_session_params_wrong_types_rejected(fields):
    from sglang.srt.managers.io_struct import SessionParams

    with pytest.raises(TypeError):
        SessionParams(**fields)


# --- caller-mm-hashes-opt-in (CVE-2026-111392) -------------------------------------------------


def test_caller_mm_hashes_need_opt_in():
    src = (SRT / "managers/tokenizer_manager.py").read_text()
    tree = ast.parse(src)
    gate = set_hash = None
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and node.value == "SGLANG_TRUST_CALLER_MM_HASHES":
            gate = node.lineno if gate is None else min(gate, node.lineno)
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "set_hash"
        ):
            set_hash = node.lineno
    assert gate is not None, "mm_hashes opt-in gate missing"
    assert set_hash is not None and gate < set_hash
    # The gate drops the caller's hashes before they can reach set_hash.
    window = src.splitlines()[gate - 1 : set_hash]
    assert any("caller_mm_hashes = None" in line for line in window)


@pytest.mark.parametrize("trusted", [False, True])
def test_preprocessed_item_hashes_need_opt_in(monkeypatch, trusted):
    import torch

    from sglang.srt.managers.schedule_batch import Modality
    from sglang.srt.multimodal.processors import base_processor as bp

    if trusted:
        monkeypatch.setenv("SGLANG_TRUST_CALLER_MM_HASHES", "1")
    else:
        monkeypatch.delenv("SGLANG_TRUST_CALLER_MM_HASHES", raising=False)
    fake = SimpleNamespace(
        ATTR_NAME_TO_MODALITY={"pixel_values": Modality.IMAGE},
        FEATURE_NAMES=["pixel_values"],
    )
    data = {
        "format": "processor_output",
        "pixel_values": torch.zeros(1, 3),
        "hash": 1234,
        "pad_value": 5678,
    }
    (item,) = bp.BaseMultimodalProcessor.collect_mm_items_from_processor_output(fake, data)
    assert item.hash == (1234 if trusted else None)
    assert item.pad_value == (5678 if trusted else None)
    # precomputed_embedding items take their hash through the same gate.
    assert bp._trusted_caller_hash(1234) == (1234 if trusted else None)
    src = Path(bp.__file__).read_text()
    assert 'hash=_trusted_caller_hash(dict_item.pop("hash", None))' in src
