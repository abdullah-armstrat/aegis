"""Build dataset E: narrated video clips with known labels, and real speech for the Whisper comparison.

Footage is from the NASA Image and Video Library (public domain) and speech from LibriSpeech
test-clean (Panayotov et al., 2015; CC BY 4.0). Media goes into the git-ignored data/E_videos/ and
data/E_speech/, manifests into data/labels/. Steps, run in order:

  speech              100 LibriSpeech utterances with their transcripts (E_speech.csv)
  footage             a window of each NASA video at 480p, without sound (E_footage.csv)
  narration           one WAV per narration line, spoken offline by the Windows built-in voice
  clips               the 34 clips (E_video_clips.csv) and their timed lines (E_video_segments.csv)
  keyframes           the middle frame of each line and a frame every 3 s, for checking the labels
  speech-tuning       100 other utterances, for tuning Whisper's decoding (E_speech_tuning.csv)
  speech-tuning-long  the tuning utterances joined into 50-60 s files (E_speech_tuning_long.csv)

The LibriSpeech archive is streamed from openslr.org, checked against the published MD5 and never
kept. The utterances picked are those with the lowest SHA-256 of "seed:utterance id" (seed 20260927;
"20260928-tuning" for the tuning set, which leaves out the first 100). Each tuning utterance is also
saved with 5 s of silence at the end. The long files use seeded (20260928) 3-8 s pauses, are filled
to 45-55 s and end with 5 s of silence. A NASA video is refused if its metadata names another
rights holder.

The clips are 15-60 s long: 15 where every line matches the picture, 15 with one or two lines that
do not (a different scene, or a wrong colour, count or place), 2 without speech (brown noise, and no
audio track) and 2 with burned-in text. Two show dataset A photos as stills. Every line was checked
against the keyframe from its own segment before its label was fixed; five were rewritten because
the keyframe did not clearly show what they said.

Run:  python backend/scripts/build_dataset_e.py <step>
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import heapq
import io
import json
import re
import subprocess
import sys
import tarfile
from datetime import date
from pathlib import Path
from urllib.parse import quote

import httpx

ROOT = Path(__file__).resolve().parents[2]
LABELS = ROOT / "data" / "labels"
SPEECH_DIR = ROOT / "data" / "E_speech"
USER_AGENT = ("aegis-dataset-builder/1.0 (University of London CM3070 student project; "
              "contact 189360416+abdullah-armstrat@users.noreply.github.com)")

LIBRISPEECH_URL = "https://www.openslr.org/resources/12/test-clean.tar.gz"
LIBRISPEECH_MD5_URL = "https://www.openslr.org/resources/12/md5sum.txt"
SPEECH_SEED = 20260927
SPEECH_COUNT = 100
TUNING_SEED = "20260928-tuning"
TUNING_SILENCE_S = 5.0
LONG_SEED = 20260928
LONG_PAUSE_S = (3.0, 8.0)
LONG_FILL_S = (45.0, 55.0)
SAMPLE_RATE = 16000


# ------------------------------------------------------------------------------ real speech
class _Stream(io.RawIOBase):
    """Read-only file over an HTTP byte stream that hashes every byte read through it."""

    def __init__(self, chunks, digest):
        self._chunks, self._digest, self._buffer = chunks, digest, b""

    def readable(self) -> bool:
        return True

    def readinto(self, target) -> int:
        while not self._buffer:
            try:
                self._buffer = next(self._chunks)
            except StopIteration:
                return 0
            self._digest.update(self._buffer)
        n = min(len(target), len(self._buffer))
        target[:n] = self._buffer[:n]
        self._buffer = self._buffer[n:]
        return n


def _duration(path: Path) -> float:
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "json", str(path)],
                         capture_output=True, text=True, check=True).stdout
    return round(float(json.loads(out)["format"]["duration"]), 3)


def build_speech(seed=SPEECH_SEED, out_dir: Path = SPEECH_DIR, manifest: str = "E_speech.csv",
                 exclude: frozenset = frozenset()) -> list[dict]:
    """Stream LibriSpeech test-clean once and keep the 100 utterances picked by the seed."""
    with httpx.Client(headers={"User-Agent": USER_AGENT}, timeout=60, follow_redirects=True) as client:
        sums = client.get(LIBRISPEECH_MD5_URL).text
        expected = next(line.split()[0] for line in sums.splitlines() if line.endswith("test-clean.tar.gz"))
        digest = hashlib.md5()  # noqa: S324 - checked against the MD5 openslr.org publishes
        chosen: list[tuple[int, str, bytes]] = []  # max-heap on the key (stored negated)
        transcripts: dict[str, str] = {}
        speakers: dict[str, str] = {}
        with client.stream("GET", LIBRISPEECH_URL, timeout=httpx.Timeout(60, read=300)) as resp:
            resp.raise_for_status()
            stream = io.BufferedReader(_Stream(resp.iter_bytes(1 << 16), digest), buffer_size=1 << 20)
            with tarfile.open(fileobj=stream, mode="r|gz") as tar:
                for member in tar:
                    name = member.name
                    if name.endswith(".flac"):
                        utt = Path(name).stem
                        if utt in exclude:
                            continue
                        key = int(hashlib.sha256(f"{seed}:{utt}".encode()).hexdigest(), 16)
                        if len(chosen) < SPEECH_COUNT or key < -chosen[0][0]:
                            item = (-key, utt, tar.extractfile(member).read())
                            if len(chosen) < SPEECH_COUNT:
                                heapq.heappush(chosen, item)
                            else:
                                heapq.heapreplace(chosen, item)
                    elif name.endswith(".trans.txt"):
                        for line in tar.extractfile(member).read().decode("utf-8").splitlines():
                            utt, _, text = line.partition(" ")
                            transcripts[utt] = text.strip()
                    elif name.endswith("SPEAKERS.TXT"):
                        for line in tar.extractfile(member).read().decode("utf-8").splitlines():
                            if line.startswith(";") or "|" not in line:
                                continue
                            fields = [f.strip() for f in line.split("|")]
                            speakers[fields[0]] = fields[1]
            for _ in stream:  # read to the end so the MD5 covers the whole archive
                pass
    if digest.hexdigest() != expected:
        sys.exit(f"LibriSpeech test-clean MD5 {digest.hexdigest()} differs from the published {expected}; "
                 "nothing was written")

    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for _, utt, data in sorted(chosen, key=lambda item: item[1]):
        path = out_dir / f"{utt}.flac"
        path.write_bytes(data)
        speaker, chapter, _ = utt.split("-")
        rows.append({
            "utterance_id": utt, "file": path.name, "speaker_id": speaker, "speaker_sex": speakers.get(speaker, ""),
            "chapter_id": chapter, "duration_s": _duration(path), "transcript": transcripts[utt],
            "sha256": hashlib.sha256(data).hexdigest(),
            "source_url": LIBRISPEECH_URL, "archive_md5": expected,
            "author": "Panayotov, Chen, Povey and Khudanpur (2015), LibriSpeech; read audiobooks from LibriVox",
            "licence": "CC BY 4.0", "licence_url": "https://creativecommons.org/licenses/by/4.0/",
            "access_date": date.today().isoformat(),
        })
    with open(LABELS / manifest, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    total = sum(r["duration_s"] for r in rows)
    print(f"LibriSpeech test-clean: MD5 matches the published {expected}; archive not kept")
    print(f"wrote {len(rows)} utterances ({total / 60:.1f} min, {len({r['speaker_id'] for r in rows})} speakers) "
          f"to {out_dir.relative_to(ROOT)} and data/labels/{manifest}")
    return rows


# ------------------------------------------------------------------------------ footage
VIDEOS = ROOT / "data" / "E_videos"
FOOTAGE_DIR = VIDEOS / "footage"
API = "https://images-api.nasa.gov"
LICENCE = "Public domain: NASA media, not subject to copyright in the United States"
LICENCE_URL = "https://www.nasa.gov/nasa-brand-center/images-and-media/"
WIDTH, HEIGHT, FPS = 854, 480, 30

# NASA source videos: key -> (NASA id, start and end second of the window kept). Picked from the
# preview frames, avoiding scenes with identifiable people. Videos with an apostrophe in the id are
# not used, because the library's search rejects them and so their rights can't be checked.
FOOTAGE = {
    "fog_vab": ("KSC-20200713-MH-JBS01-0001-Creative_Videography_Fog_Rolling_Over_VAB_Timelapse-3254205", 0, 101),
    "eagles": ("KSC-20260313-MH-JBS01-0001-Wildlife_Video_Bald_Eagles_BROLL-M19993", 0, 148),
    "pad_timelapse": ("KSC-20190826-MH-KLS01_0001-LH2_Time_Lapse_of_construction_at_39B-3223721", 0, 150),
    "pad_sunset": ("SLS_KSC_01282026_ART II at Pad sunset anamorphic", 0, 135),
    "moon_sunrise": ("KSC-20260201-MH-JBS01-0001-Artemis_II_Full_Moon_Sunrise_Timelapse-M18703", 0, 41),
    "shoreline": ("KSC-20190524-MH-MTD01-0001-Drone_Footage_Shoreline_Beach_House_VAB_Pad_41_Pad_39A-3222188", 0, 125),
    "dorian": ("jsc2019m000806_Hurricane_Dorian_190902", 0, 192),
    "milton": ("jsc2024m000174_Cameras_On_The_International_Space_Station_Capture_New_Views_Of_Hurricane_Milton-241009", 0, 180),
    "snow": ("GSFC_20120703_GPM_m11067_Snow_montage", 0, 79),
    "barge_river": ("MSFC_08212024_LVSA II shipping on Pegasus Decatur aerials", 0, 191),
    "barge_arrival": ("KSC-20210427-MH-MTD01-0001_Artemis_I_Core_Stage_Arrival_at_KSC_DRONE-3274757", 0, 175),
    "rollout_drone": ("KSC-20211014-MH-MTD01-0001-Lucy_Rollout_to_Pad_SLC-41_DRONE-3289002", 0, 159),
    "helicopters": ("KSC-20200930-MH-FJM01_0001-New_NASA_Helicopters_Arrive_At_KSC-3259552", 0, 205),
    "global_hawk": ("ARC-20130816-AAV3521-Drone-Returns-After-Studying-Atlantic-Storms", 0, 130),
    "helo_sunrise": ("KSC-20221114-MH-JBP01-Artemis_I_Sunrise_Helo_Shots-WON_3318261", 0, 240),
    "rollout_sunset": ("SLS_KSC_Artemis II Rollout to pad 1172026_broll_timelapse_sunset", 0, 240),
    "aerial_survey": ("KSC-20190905-MH-CMS01_0001-DART_Support_for_Hurricane_Dorian-3230374", 0, 170),
}
RIGHTS = re.compile(r"copyright|rights|credit", re.I)
THIRD_PARTY = re.compile(r"©|courtesy|all rights reserved", re.I)


def _nasa_video(client: httpx.Client, nasa_id: str) -> dict:
    """Get a video's NASA record and 720p file URL; exit if anyone else seems to hold rights."""
    items = client.get(f"{API}/search", params={"nasa_id": nasa_id}).json()["collection"]["items"]
    record = next(i["data"][0] for i in items if i["data"][0]["nasa_id"] == nasa_id)
    meta = client.get(client.get(f"{API}/metadata/{quote(nasa_id)}").json()["location"]).json()
    for key, value in meta.items():
        value = str(value)
        if THIRD_PARTY.search(value) or (RIGHTS.search(key) and value.strip() and "nasa" not in value.lower()):
            sys.exit(f"{nasa_id}: possible third-party rights in {key}: {value[:80]!r}")
    if THIRD_PARTY.search(str(record.get("description", ""))):
        sys.exit(f"{nasa_id}: possible third-party rights in the description")
    assets = [i["href"] for i in client.get(f"{API}/asset/{quote(nasa_id)}").json()["collection"]["items"]]
    medium = next(a for a in assets if a.endswith("~medium.mp4"))
    return {"record": record, "url": medium.replace("http://", "https://"),
            "credit": str(meta.get("XMP:Credit", "")).strip()}


def build_footage() -> None:
    """Save a silent 480p copy of each source window, read straight from NASA's 720p file."""
    FOOTAGE_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    with httpx.Client(headers={"User-Agent": USER_AGENT}, timeout=60, follow_redirects=True) as client:
        for key, (nasa_id, start, end) in FOOTAGE.items():
            got = _nasa_video(client, nasa_id)
            out = FOOTAGE_DIR / f"{key}.mp4"
            if not out.exists():
                subprocess.run([
                    "ffmpeg", "-v", "error", "-y", "-user_agent", USER_AGENT, "-ss", str(start), "-to", str(end),
                    "-i", got["url"], "-an",
                    "-vf", f"scale={WIDTH}:{HEIGHT}:force_original_aspect_ratio=decrease,"
                           f"pad={WIDTH}:{HEIGHT}:(ow-iw)/2:(oh-ih)/2,fps={FPS},format=yuv420p",
                    "-c:v", "libx264", "-preset", "veryfast", "-crf", "23", str(out)], check=True)
            record = got["record"]
            rows.append({
                "footage": key, "file": out.name, "nasa_id": nasa_id, "window_start_s": start, "window_end_s": end,
                "duration_s": _duration(out), "date_created": str(record["date_created"])[:10],
                "source_url": f"https://images.nasa.gov/details/{quote(nasa_id)}", "file_url": got["url"],
                "author": "; ".join(p for p in (f"NASA {record.get('center', '')}".strip(), got["credit"]) if p),
                "licence": LICENCE, "licence_url": LICENCE_URL,
                "title": " ".join(str(record.get("title", "")).split()),
                "nasa_description": " ".join(str(record.get("description", "")).split()),
                "access_date": date.today().isoformat(),
            })
            print(f"  {key:<15} {rows[-1]['duration_s']:>7.1f} s  {nasa_id[:70]}")
    with open(LABELS / "E_footage.csv", "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} footage files to {FOOTAGE_DIR.relative_to(ROOT)} and data/labels/E_footage.csv")


# ------------------------------------------------------------------------------ clips
NARRATION_DIR = VIDEOS / "narration"
CLIPS_DIR = VIDEOS / "clips"
KEYFRAMES_DIR = VIDEOS / "keyframes"
A_DIR = ROOT / "data" / "A_originals"
VOICE = "Microsoft Zira Desktop"  # same voice for every clip, so the voice can't give the label away
SAMPLE_RATE = 22050
FONT = "C\\:/Windows/Fonts/arialbd.ttf"
MIN_GAP_S = 0.5

M, D, W = "matches", "different_scene", "wrong_detail"

# Each clip: id, type, source (footage key, or a dataset A file shown as a still), source start and
# end in seconds, topic, burned-in text, audio when there is no speech, and the narration lines as
# (start second in the clip, text, label, what is wrong if it is a wrong detail).
CLIPS = [
    ("E01", "narrated_match", "fog_vab", 11, 56, "weather", "", "", [
        (1, "Thick fog hides a tall building.", M, ""),
        (8, "The fog lifts and the tall building comes into view.", M, ""),
        (20, "A large flag is painted on the side of the building.", M, ""),
        (32, "Green grass stretches in front of it under a blue sky.", M, "")]),
    ("E02", "narrated_mismatch", "pad_timelapse", 20, 65, "transport", "", "", [
        (1, "A tall crane works on a building site.", M, ""),
        (10, "Behind the site, the sea runs along the horizon.", M, ""),
        (20, "A herd of cows grazes in a muddy field.", D, ""),
        (32, "A large round grey tank rises on concrete legs.", M, "")]),
    ("E03", "narrated_match", "moon_sunrise", 6, 28, "weather", "", "", [
        (1, "At night, a tall rocket stands lit up on its launch pad.", M, ""),
        (8, "A bright full moon sinks slowly beside the rocket.", M, ""),
        (15, "Lights on the pad reflect in the water below.", M, "")]),
    ("E04", "narrated_match", "eagles", 87, 117, "animals", "", "", [
        (1, "Two bald eagles perch high in a bare tree.", M, ""),
        (12, "A huge nest of sticks sits in the branches.", M, ""),
        (24, "Small dark chicks move about inside the nest.", M, "")]),
    ("E05", "narrated_mismatch", "eagles", 36, 84, "animals", "", "", [
        (1, "An eagle with a white head watches over two chicks.", M, ""),
        (12, "A young eagle stretches its dark wings above the nest.", M, ""),
        (24, "A red fox runs across a snowy field.", D, ""),
        (36, "The young eagle flaps its wings against a white sky.", M, "")]),
    ("E06", "narrated_match", "shoreline", 6, 46, "landscapes", "", "", [
        (1, "A drone flies over a small blue house among green bushes.", M, ""),
        (9, "A tall white tower stands in the distance.", M, ""),
        (20, "Waves break on a wide sandy beach.", M, ""),
        (30, "Behind the beach, the house sits on low dunes.", M, "")]),
    ("E07", "narrated_mismatch", "shoreline", 76, 124, "landscapes", "", "", [
        (1, "A curving sandy path leads to a blue house.", M, ""),
        (10, "A red tractor ploughs a field of wheat.", D, ""),
        (22, "Green scrubland spreads towards a tall white tower.", M, ""),
        (34, "This beach house stands on the coast of Maine.", W, "place: the house is at Kennedy Space Center, Florida")]),
    ("E08", "narrated_match", "dorian", 6, 60, "weather", "", "", [
        (1, "From orbit, a huge hurricane swirls over the ocean.", M, ""),
        (12, "Parts of the space station stand out as dark shapes.", M, ""),
        (24, "The storm's eye is a small dark hole in the clouds.", M, ""),
        (38, "Thick white cloud bands spiral around the centre.", M, "")]),
    ("E09", "narrated_mismatch", "milton", 0, 60, "weather", "", "", [
        (1, "A storm covers the ocean below the space station.", M, ""),
        (12, "A long robotic arm reaches out from the station.", M, ""),
        (24, "A crowded market sells fruit and vegetables.", D, ""),
        (36, "Solar panels stretch across the lower left of the view.", M, ""),
        (48, "Three hurricanes line up across the sea.", W, "count: one storm is shown")]),
    ("E10", "narrated_match", "snow", 18, 41, "weather", "", "", [
        (1, "Snow settles on the leaves of a thin branch.", M, ""),
        (7, "Snowflakes drift down past the branch.", M, ""),
        (15, "A truck with a snowplough pushes snow along a street.", M, "")]),
    ("E11", "narrated_mismatch", "snow", 42, 78, "weather", "", "", [
        (1, "Bare trees stand along the edge of a calm lake.", M, ""),
        (10, "A busy motorway is full of cars at rush hour.", D, ""),
        (25, "The low sun shines through the trees onto the snow.", M, "")]),
    ("E12", "narrated_match", "barge_river", 20, 50, "transport", "", "", [
        (1, "A long white barge passes under a steel bridge.", M, ""),
        (10, "Traffic crosses the bridge above the river.", M, ""),
        (20, "The barge moves along a wide river with green banks.", M, "")]),
    ("E13", "narrated_mismatch", "barge_river", 105, 150, "transport", "", "", [
        (1, "A barge slides past a tall lift bridge on the river.", M, ""),
        (10, "Sunlight glitters on the water around the barge.", M, ""),
        (20, "Snow-capped mountains rise above a frozen lake.", D, ""),
        (32, "Small boats float near the far shore of the river.", M, "")]),
    ("E14", "narrated_match", "barge_arrival", 6, 40, "transport", "", "", [
        (1, "A tugboat pushes a covered barge along a waterway.", M, ""),
        (9, "A small white boat travels ahead of the barge.", M, ""),
        (19, "The barge carries a long white cover with a curved roof.", M, ""),
        (27, "Green marsh lines both sides of the water.", M, "")]),
    ("E15", "narrated_mismatch", "barge_arrival", 105, 165, "transport", "", "", [
        (1, "The barge approaches a huge square building.", M, ""),
        (12, "A bright green tugboat pushes the barge from behind.", W, "colour: the tugboat is red and white"),
        (24, "A cruise ship full of passengers docks in the harbour.", D, ""),
        (40, "The giant building has a round blue logo near the top.", M, "")]),
    ("E16", "narrated_match", "rollout_drone", 9, 54, "transport", "", "", [
        (1, "A tall white building with a flag stands near a lagoon.", M, ""),
        (12, "Green marsh and water stretch out behind it.", M, ""),
        (30, "A rocket now stands beside the tall building.", M, ""),
        (38, "Clouds drift across a pale blue sky.", M, "")]),
    ("E17", "narrated_mismatch", "rollout_drone", 99, 158, "transport", "", "", [
        (1, "A rocket stands on a launch pad between tall towers.", M, ""),
        (12, "A large white sphere sits near the pad.", M, ""),
        (24, "Tall lightning towers surround the rocket.", M, ""),
        (36, "The rocket on the pad is painted bright green.", W, "colour: the rocket is white"),
        (48, "A flock of sheep crosses a country lane.", D, "")]),
    ("E18", "narrated_match", "helicopters", 50, 95, "transport", "", "", [
        (1, "A white helicopter hovers above green trees.", M, ""),
        (12, "It slowly comes down towards a paved area.", M, ""),
        (24, "The helicopter has a blue stripe along its side.", M, ""),
        (36, "It settles close to the ground in front of the trees.", M, "")]),
    ("E19", "narrated_mismatch", "helicopters", 12, 48, "transport", "", "", [
        (1, "A large white hangar stands under a blue sky.", M, ""),
        (12, "A pod of whales swims beside a sailing boat.", D, ""),
        (24, "A small helicopter appears far away above the trees.", M, "")]),
    ("E20", "narrated_match", "global_hawk", 0, 25, "transport", "", "", [
        (1, "A white aircraft approaches from a hazy sky.", M, ""),
        (9, "It descends towards a runway lined with trees.", M, ""),
        (16, "The long-winged plane touches down on the runway.", M, "")]),
    ("E21", "narrated_mismatch", "global_hawk", 30, 63, "transport", "", "", [
        (1, "A white plane with very long wings rolls along a runway.", M, ""),
        (10, "Green grass lines the edges of the runway.", M, ""),
        (19, "A crowd of children plays in a school playground.", D, ""),
        (27, "An orange plane with long wings rolls along the runway.", W, "colour: the plane is white")]),
    ("E22", "narrated_match", "helo_sunrise", 12, 60, "transport", "", "", [
        (1, "In the dark, a tall orange rocket stands on its pad.", M, ""),
        (14, "Bright lights shine around the base of the launch tower.", M, ""),
        (28, "The rocket glows orange and white against the dark sky.", M, "")]),
    ("E23", "narrated_mismatch", "helo_sunrise", 123, 172, "transport", "", "", [
        (1, "At dawn, a rocket stands beside its launch tower.", M, ""),
        (12, "The rocket has a white top and an orange body.", M, ""),
        (24, "A snowstorm buries a quiet mountain village.", D, ""),
        (36, "The sky behind the rocket turns deep blue.", M, "")]),
    ("E24", "narrated_match", "aerial_survey", 96, 146, "landscapes", "", "", [
        (1, "Waves roll onto a sandy beach seen from above.", M, ""),
        (14, "Green plants grow behind the sand.", M, ""),
        (28, "White foam spreads along the edge of the water.", M, ""),
        (40, "A long strip of sand runs beside the waves.", M, "")]),
    ("E25", "narrated_mismatch", "aerial_survey", 6, 47, "landscapes", "", "", [
        (1, "A drone circles a huge grey and white building.", M, ""),
        (12, "A giant flag and a round logo decorate its side.", M, ""),
        (24, "Camels walk across golden sand dunes.", D, ""),
        (35, "Equipment and materials cover a building site nearby.", M, "")]),
    ("E26", "narrated_match", "pad_sunset", 15, 45, "weather", "", "", [
        (1, "Tall towers stand in silhouette against the setting sun.", M, ""),
        (12, "The sun flares brightly behind the rocket.", M, ""),
        (22, "A round water tower stands to the right.", M, "")]),
    ("E27", "narrated_mismatch", "rollout_sunset", 72, 96, "transport", "", "", [
        (1, "A giant rocket rides on a huge tracked vehicle.", M, ""),
        (12, "A large flag hangs on the side of the vehicle.", M, ""),
        (18, "Seagulls dive into a stormy sea.", D, "")]),
    ("E28", "narrated_match", "A15_KSC-98PC-572.jpg", 0, 20, "transport", "", "", [
        (1, "A red locomotive sits on a railway track.", M, ""),
        (7, "A long line of freight cars stretches behind it.", M, ""),
        (13, "White clouds float in a blue sky above the train.", M, "")]),
    ("E29", "narrated_mismatch", "A28_KSC-2010-5875.jpg", 0, 20, "animals", "", "", [
        (1, "A white bird with a dark head wades in calm water.", M, ""),
        (7, "Its long curved beak is bright yellow.", W, "colour: the beak is dark grey"),
        (13, "Ripples spread across the water around its legs.", M, "")]),
    ("E30", "narrated_mismatch", "dorian", 123, 183, "weather", "", "", [
        (1, "A large part of the space station fills the left of the view.", M, ""),
        (14, "Beyond it, a hurricane swirls over the ocean.", M, ""),
        (28, "A marching band parades down a city street.", D, ""),
        (42, "The hurricane's eye sits over the deserts of Arizona.", W, "place: the storm is over the northwestern Bahamas")]),
    ("E31", "no_speech", "aerial_survey", 48, 82, "landscapes", "", "brown_noise", []),
    ("E32", "no_speech", "milton", 81, 118, "weather", "", "none", []),
    ("E33", "on_screen_text", "pad_sunset", 84, 107, "weather", "SUNSET AT THE LAUNCH PAD", "", [
        (1, "The sun sets behind tall grass and a distant rocket.", M, ""),
        (10, "Clouds glow orange above the horizon.", M, "")]),
    ("E34", "on_screen_text", "barge_river", 60, 80, "transport", "BARGE ON THE RIVER", "", [
        (1, "A covered barge travels up a wide river.", M, ""),
        (9, "A small boat leaves a white wake beside it.", M, "")]),
]
STATIC = {"E01", "E02", "E03", "E28", "E29"}  # fixed camera for the whole clip

TTS_SCRIPT = r"""
param([string]$List)
Add-Type -AssemblyName System.Speech
$s = New-Object System.Speech.Synthesis.SpeechSynthesizer
$s.SelectVoice('__VOICE__')
$s.Rate = 0
$fmt = New-Object System.Speech.AudioFormat.SpeechAudioFormatInfo(__RATE__, [System.Speech.AudioFormat.AudioBitsPerSample]::Sixteen, [System.Speech.AudioFormat.AudioChannel]::Mono)
foreach ($row in Get-Content -Encoding UTF8 $List) {
    $parts = $row -split "`t", 2
    $s.SetOutputToWaveFile($parts[0], $fmt)
    $s.Speak($parts[1])
    $s.SetOutputToNull()
}
$s.Dispose()
"""


def build_narration() -> None:
    """Write one WAV per narration line, spoken offline by the Windows built-in voice."""
    import tempfile
    import wave

    NARRATION_DIR.mkdir(parents=True, exist_ok=True)
    rows = [(NARRATION_DIR / f"{cid}_{n:02d}.wav", text)
            for cid, _, _, _, _, _, _, _, lines in CLIPS for n, (_, text, _, _) in enumerate(lines, 1)]
    with tempfile.TemporaryDirectory() as tmp:
        ps1, listing = Path(tmp) / "speak.ps1", Path(tmp) / "lines.tsv"
        ps1.write_text(TTS_SCRIPT.replace("__VOICE__", VOICE).replace("__RATE__", str(SAMPLE_RATE)), encoding="utf-8")
        listing.write_text("\n".join(f"{p}\t{t}" for p, t in rows) + "\n", encoding="utf-8")
        subprocess.run(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(ps1), str(listing)],
                       check=True)
    for path, _ in rows:
        with wave.open(str(path)) as w:
            assert w.getframerate() == SAMPLE_RATE and w.getnchannels() == 1 and w.getnframes() > 0, path
    print(f"wrote {len(rows)} narration lines in the voice {VOICE!r} to {NARRATION_DIR.relative_to(ROOT)}")


def _read_wav(path: Path):
    import wave

    import numpy as np

    with wave.open(str(path)) as w:
        return np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)


def _timed_lines(cid: str, lines, duration: float) -> list[dict]:
    """Start and end of each spoken line; exits if lines overlap or speech runs past the clip's end."""
    out = []
    for n, (start, text, label, note) in enumerate(lines, 1):
        samples = _read_wav(NARRATION_DIR / f"{cid}_{n:02d}.wav")
        end = round(start + len(samples) / SAMPLE_RATE, 3)
        if out and start < out[-1]["end_s"] + MIN_GAP_S:
            sys.exit(f"{cid} line {n} starts before line {n - 1} has finished")
        if end > duration - MIN_GAP_S:
            sys.exit(f"{cid} line {n} runs past the end of the clip")
        out.append({"index": n, "start_s": float(start), "end_s": end, "text": text, "label": label,
                    "note": note, "samples": samples})
    return out


def build_clips() -> None:
    """Cut, narrate and encode every clip; write the clip and line manifests."""
    import tempfile
    import wave

    import numpy as np

    CLIPS_DIR.mkdir(parents=True, exist_ok=True)
    footage = {r["footage"]: r for r in csv.DictReader(open(LABELS / "E_footage.csv", encoding="utf-8"))}
    a_rows = {r["file"]: r for r in csv.DictReader(open(LABELS / "A_originals.csv", encoding="utf-8"))}
    clip_rows, line_rows = [], []
    for cid, kind, source, start, end, topic, text_overlay, silent_audio, lines in CLIPS:
        duration = end - start
        if not 15 <= duration <= 60:
            sys.exit(f"{cid}: {duration} s is outside 15-60 s")
        timed = _timed_lines(cid, lines, duration)
        out = CLIPS_DIR / f"{cid}.mp4"
        vf = f"scale={WIDTH}:{HEIGHT}:force_original_aspect_ratio=decrease,pad={WIDTH}:{HEIGHT}:(ow-iw)/2:(oh-ih)/2,fps={FPS}"
        if text_overlay:
            vf += (f",drawtext=fontfile='{FONT}':text='{text_overlay}':fontsize=40:fontcolor=white:"
                   "box=1:boxcolor=black@0.6:boxborderw=14:x=(w-text_w)/2:y=h-text_h-40")
        vf += ",format=yuv420p"
        if source in a_rows:
            video_in = ["-loop", "1", "-t", str(duration), "-i", str(A_DIR / source)]
        else:
            video_in = ["-ss", str(start), "-t", str(duration), "-i", str(FOOTAGE_DIR / footage[source]["file"])]
        with tempfile.TemporaryDirectory() as tmp:
            if timed:
                track = np.zeros(int(round(duration * SAMPLE_RATE)), dtype=np.int16)
                for line in timed:
                    at = int(round(line["start_s"] * SAMPLE_RATE))
                    track[at:at + len(line["samples"])] = line["samples"]
                wav = Path(tmp) / "narration.wav"
                with wave.open(str(wav), "wb") as w:
                    w.setnchannels(1)
                    w.setsampwidth(2)
                    w.setframerate(SAMPLE_RATE)
                    w.writeframes(track.tobytes())
                audio_in, audio_out = ["-i", str(wav)], ["-c:a", "aac", "-b:a", "96k", "-ar", "44100"]
            elif silent_audio == "brown_noise":
                audio_in = ["-f", "lavfi", "-t", str(duration), "-i", "anoisesrc=color=brown:amplitude=0.05:seed=7"]
                audio_out = ["-c:a", "aac", "-b:a", "96k", "-ar", "44100"]
            else:
                audio_in, audio_out = [], ["-an"]
            subprocess.run(["ffmpeg", "-v", "error", "-y", *video_in, *audio_in, "-map", "0:v:0",
                            *(["-map", "1:a:0"] if audio_in else []), "-vf", vf, "-t", str(duration),
                            "-c:v", "libx264", "-preset", "veryfast", "-crf", "23", *audio_out, str(out)], check=True)
        is_a = source in a_rows
        src = a_rows[source] if is_a else footage[source]
        clip_rows.append({
            "clip_id": cid, "file": out.name, "type": kind, "duration_s": _duration(out),
            "has_speech": "yes" if timed else "no",
            "audio": "narration" if timed else ("brown noise, no speech" if silent_audio == "brown_noise" else "no audio track"),
            "has_on_screen_text": "yes" if text_overlay else "no", "on_screen_text": text_overlay,
            "static_shot": "yes" if cid in STATIC else "no",
            "contains_image_from_A": source if is_a else "",
            "source": source, "source_nasa_id": src["nasa_id"],
            "source_start_s": start if not is_a else "", "source_end_s": end if not is_a else "",
            "source_url": src["source_url"], "licence": src["licence"],
            "topic": topic, "n_lines": len(timed), "n_mismatched": sum(t["label"] != M for t in timed),
            "voice": VOICE if timed else "", "script": " ".join(t["text"] for t in timed),
        })
        for t in timed:
            line_rows.append({
                "clip_id": cid, "line_index": t["index"], "start_s": t["start_s"], "end_s": t["end_s"],
                "text": t["text"], "label": t["label"], "wrong_detail": t["note"],
                "keyframe_s": round((t["start_s"] + t["end_s"]) / 2, 2),
            })
    for name, rows in (("E_video_clips.csv", clip_rows), ("E_video_segments.csv", line_rows)):
        with open(LABELS / name, "w", encoding="utf-8", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    print(f"wrote {len(clip_rows)} clips to {CLIPS_DIR.relative_to(ROOT)}; {len(line_rows)} lines "
          f"({sum(r['label'] == M for r in line_rows)} match, {sum(r['label'] == D for r in line_rows)} different scene, "
          f"{sum(r['label'] == W for r in line_rows)} wrong detail)")


def build_keyframes() -> None:
    """Save the middle frame of each line and a frame every 3 s, for checking the labels."""
    KEYFRAMES_DIR.mkdir(parents=True, exist_ok=True)
    for row in csv.DictReader(open(LABELS / "E_video_segments.csv", encoding="utf-8")):
        out = KEYFRAMES_DIR / f"{row['clip_id']}_line{int(row['line_index']):02d}.jpg"
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", row["keyframe_s"], "-i",
                        str(CLIPS_DIR / f"{row['clip_id']}.mp4"), "-frames:v", "1", str(out)], check=True)
    for row in csv.DictReader(open(LABELS / "E_video_clips.csv", encoding="utf-8")):
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(CLIPS_DIR / row["file"]), "-vf", "fps=1/3,scale=284:160",
                        str(KEYFRAMES_DIR / f"{row['clip_id']}_every3s_%02d.jpg")], check=True)
    print(f"wrote keyframes to {KEYFRAMES_DIR.relative_to(ROOT)}")


def build_speech_tuning() -> None:
    """Build the tuning set: 100 other utterances, each also saved with 5 s of silence at the end."""
    with open(LABELS / "E_speech.csv", encoding="utf-8") as fh:
        originals = frozenset(r["utterance_id"] for r in csv.DictReader(fh))
    out_dir = SPEECH_DIR / "tuning"
    rows = build_speech(TUNING_SEED, out_dir, "E_speech_tuning.csv", originals)
    for r in rows:
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(out_dir / r["file"]), "-af",
                        f"apad=pad_dur={TUNING_SILENCE_S}", "-ar", "16000", "-ac", "1",
                        str(out_dir / f"{r['utterance_id']}_silence.wav")], check=True)
    print(f"added {TUNING_SILENCE_S:.0f} s of silence to the end of each, as *_silence.wav")


def build_speech_tuning_long() -> None:
    """Join the tuning utterances into long files with pauses and 5 s of silence at the end.

    They match the length of the clips where Whisper invented sentences in the silence at the end.
    """
    import random
    import wave

    import numpy as np
    from whisper.audio import load_audio

    with open(LABELS / "E_speech_tuning.csv", encoding="utf-8") as fh:
        rows = {r["utterance_id"]: r for r in csv.DictReader(fh)}
    rng = random.Random(LONG_SEED)
    remaining = sorted(rows)
    rng.shuffle(remaining)
    audio = {u: load_audio(str(SPEECH_DIR / "tuning" / rows[u]["file"])) for u in remaining}
    files = []
    # Fill each file, in shuffled order, with whatever still fits under 55 s; stop once one can't reach 45 s.
    while True:
        parts, length = [], 0.0
        for u in list(remaining):
            pause = round(rng.uniform(*LONG_PAUSE_S), 1) if parts else 0.0
            dur = len(audio[u]) / SAMPLE_RATE
            if length + pause + dur <= LONG_FILL_S[1]:
                parts.append((u, pause))
                length += pause + dur
        if length < LONG_FILL_S[0]:
            break
        for u, _ in parts:
            remaining.remove(u)
        files.append(parts)
    out_dir = SPEECH_DIR / "tuning_long"
    out_dir.mkdir(parents=True, exist_ok=True)
    plan = []
    for n, parts in enumerate(files, 1):
        pieces = []
        for u, pause in parts:
            pieces += [np.zeros(int(round(pause * SAMPLE_RATE)), dtype=np.float32), audio[u]]
        pieces.append(np.zeros(int(TUNING_SILENCE_S * SAMPLE_RATE), dtype=np.float32))
        signal = np.concatenate(pieces)
        name = f"long_{n:02d}.wav"
        with wave.open(str(out_dir / name), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(SAMPLE_RATE)
            w.writeframes((np.clip(signal, -1, 1) * 32767).astype("<i2").tobytes())
        plan.append({"file_id": f"long_{n:02d}", "file": name, "duration_s": round(len(signal) / SAMPLE_RATE, 2),
                     "utterances": " ".join(u for u, _ in parts),
                     "pauses_s": " ".join(f"{p:.1f}" for _, p in parts[1:]),
                     "transcript": " ".join(rows[u]["transcript"] for u, _ in parts)})
    with open(LABELS / "E_speech_tuning_long.csv", "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(plan[0]))
        w.writeheader()
        w.writerows(plan)
    used = sum(len(p) for p in files)
    print(f"wrote {len(plan)} files of {min(r['duration_s'] for r in plan)}-{max(r['duration_s'] for r in plan)} s "
          f"from {used} utterances ({len(remaining)} left out) to {out_dir.relative_to(ROOT)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("step", choices=["speech", "speech-tuning", "speech-tuning-long", "footage", "narration",
                                         "clips", "keyframes"])
    args = parser.parse_args()
    {"speech": build_speech, "speech-tuning": build_speech_tuning, "speech-tuning-long": build_speech_tuning_long,
     "footage": build_footage,
     "narration": build_narration, "clips": build_clips, "keyframes": build_keyframes}[args.step]()


if __name__ == "__main__":
    main()
