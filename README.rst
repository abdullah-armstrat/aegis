=====
Aegis
=====

Aegis checks an image or video post and its caption before you share it. It runs a set of
checks and reports each finding with the evidence behind it, a plain-English explanation, and
what you can check yourself. It never says whether a post is true or false: it shows you what is
worth checking.

It was built for the University of London BSc Computer Science final project (CM3070), on the
brief "Orchestrating AI models to achieve a goal".

.. contents::
   :local:
   :depth: 1


What it does
============

You give it a picture or a short video (up to 60 seconds), the caption as posted, and, if you
know it, the date the post claims. Each check comes back as one of: *Flag raised*, *No issue
found*, or *Couldn't check* (with the reason). The checks are:

Recycled context
    Has this picture been online before? The picture, or each keyframe of a video, is matched by
    content against an offline image history index: a perceptual hash first, then keypoint
    matching for cropped, bordered and screenshot-framed copies. If you give the posting date,
    an earlier copy dated before it is flagged. For images you can also tick "search the web",
    which sends the picture to Google Cloud Vision (off unless a key is set on the server).

Caption vs picture
    Does the picture show the kind of scene the caption describes? CLIP compares the picture
    (or each keyframe) with the caption.

Shouting style
    Is the caption written to pressure you: many words in capitals, several exclamation marks,
    urgency words, set phrases? This is about how it is written, not what it says.

Speech vs picture (video)
    Whisper transcribes the speech; CLIP compares each stretch of speech with the picture at that
    moment. A weak match is flagged with its time, and the player jumps there.

A second opinion from a local language model (optional)
    If Ollama is running with phi3:mini and the LLM is switched on, the model reads the caption,
    a description of the picture and any text in it, and says whether they are about the same
    thing. It is shown as a separate finding, below the others.

What it does not do
-------------------

* It gives no verdict and no score for the whole post.
* It does not check facts. It cannot tell whether a name, place, date or number is right; the
  caption and speech checks only compare the kind of scene.
* It does not detect AI-generated or edited pictures.
* The offline index is small (40 dated NASA photos and 2 made-up examples), so "no earlier copy
  found" means little on its own. The web search finds more, but a page's date is often later
  than the picture's first appearance.
* It reads English only.


Requirements
============

Developed and tested on Windows 11 with Python 3.12, on a CPU-only laptop with 8 GB of RAM.

* Python 3.12
* Tesseract OCR. On Windows the app looks in ``C:\Program Files\Tesseract-OCR\tesseract.exe``;
  elsewhere set ``AEGIS_TESSERACT_CMD`` to the program's path, or leave it empty to use ``PATH``.
* ffmpeg on ``PATH`` (for video)
* Node.js 20 or later (for the web interface)
* Google Chrome (for the browser tests and the interface review)
* Optional: Ollama with ``phi3:mini`` (``ollama pull phi3:mini``) for the language model


Setup
=====

From the repository root (the first ``pip`` line installs the CPU-only builds of PyTorch that
Aegis was tested with)::

    cd backend
    python -m venv .venv
    .venv\Scripts\activate            (Windows)
    source .venv/bin/activate         (macOS or Linux)
    pip install torch==2.12.0 torchvision==0.27.0 --index-url https://download.pytorch.org/whl/cpu
    pip install -r requirements.txt

    cd ../frontend
    npm install

Models
------

The app never downloads a model while it runs: it loads them from files already on the machine,
and a check whose model is missing comes back *Couldn't check* with the reason. Fetch them once,
with the virtual environment active, from ``backend/``::

    python scripts/probe_models.py --download
    python -c "from huggingface_hub import snapshot_download; snapshot_download('Salesforce/blip-image-captioning-base')"

The first command fetches CLIP (ViT-B/32, RN50, ViT-B/16) and Whisper (tiny, base) from OpenAI's
servers; the spaCy model was installed by ``pip``. The second fetches the BLIP captioner, which is
only used by the older caption checks and the language model. ``python scripts/probe_models.py``
then loads each model with the network blocked, to show they are all local.

Settings
--------

Settings come from environment variables starting with ``AEGIS_``, or from ``backend/.env``;
``backend/.env.example`` lists every one with its default. The defaults are the shipped
configuration: offline, the picture-only caption check, the language model off, no web search.
The Google Cloud Vision key is read only from the environment variable
``GOOGLE_VISION_API_KEY``; never put it in a file in the repository.


Running it
==========

In one terminal, from ``backend/`` with the virtual environment active::

    python -m uvicorn app.main:app --port 8000

In another, from ``frontend/``::

    npm run dev

Then open http://localhost:5173. With no key and no Ollama this is the default mode: every check
runs offline, and the web-search box is not offered.

To turn the language model on, start Ollama and set ``AEGIS_USE_LLM=true`` before starting the
backend. To offer the web search, set ``GOOGLE_VISION_API_KEY`` in the backend's environment.

The API is documented at http://localhost:8000/docs (``POST /analyze`` for images,
``POST /analyze/video`` for videos, ``GET /health``).


Tests
=====

Backend, from ``backend/`` with the virtual environment active::

    python -m pytest                 all tests
    python -m pytest -m "not slow"   without the ones that load BLIP or CLIP

Browser tests, from ``frontend/`` (they start the backend and the web interface themselves, and
need Google Chrome)::

    npx playwright test

The production build of the web interface::

    npm run build


The evaluation
==============

One command regenerates every number and chart the report uses, from the repository root with
the virtual environment active::

    python backend/scripts/run_final_eval.py

It writes one JSON file per step to ``results/``, every figure the report may cite to
``results/report_numbers.csv`` (with its interval, and the file and step it comes from), and the
charts to ``results/figures/`` (PNG, 300 dpi). ``--help`` lists the steps; name some to run only
those.

Slow model outputs (BLIP descriptions, OCR text, CLIP and spaCy scores, the language model's
answers, Whisper's transcripts and the video path's evidence) are kept in
``data/final_eval_cache/`` and replayed on later runs. ``--rebuild`` deletes that cache and builds
it again from the models. Everything else runs live each time.

The evaluation needs the datasets below, Ollama with ``phi3:mini`` running, and the frontend
installed (for the interface review). On the development laptop, building the cache from scratch
took about four and a half hours, most of it the language model; a run that replays the cache
takes about an hour.

The results in ``results/`` are the frozen final run. Timings, memory figures and the interface
review's measured waits change from run to run; every other number is reproduced exactly when the
cache is replayed.


Datasets
========

Only identifiers, labels, checksums and the scripts that fetch the data are in the repository
(``data/labels/``). Pictures, videos, audio and texts stay on the machine that fetched them. Run
each script from the repository root with the virtual environment active.

.. list-table::
   :header-rows: 1
   :widths: 20 40 40

   * - Dataset
     - How to fetch it
     - Source and licence
   * - A: 40 dated photos
     - ``python backend/scripts/get_dataset_a.py``, then
       ``python backend/scripts/build_image_index.py`` to rebuild the index from them
     - NASA Image and Video Library; public domain (items naming another rights holder were refused)
   * - Hard pairs: 20 pairs of photos of one subject
     - ``python backend/scripts/get_hard_pairs.py``
     - NASA Image and Video Library; public domain
   * - C: VERITE, 300 sampled pairs and the fresh set
     - ``python backend/scripts/build_verite_sample.py``, then the same with ``--fresh``
     - Annotations from the VERITE authors' repository (Papadopoulos et al., 2024; Apache-2.0).
       The images are downloaded from their original web addresses and belong to their owners;
       they are not shared here
   * - D: the author's photos
     - Not published
     - The author's own photos, taken for the project and never posted; only aggregate results
       are in ``results/``
   * - E: 34 narrated clips, and real speech
     - ``python backend/scripts/build_dataset_e.py STEP`` with the steps ``footage``,
       ``narration``, ``clips``, ``speech``, ``speech-tuning`` and ``speech-tuning-long`` in turn
     - NASA footage (public domain), narrated by the Windows voice Microsoft Zira; LibriSpeech
       test-clean (Panayotov et al., 2015; CC BY 4.0)
   * - F: 99 meme texts
     - ``python backend/scripts/get_dataset_f.py``
     - SemEval-2021 Task 6 (Dimitrov et al., 2021; "free for general research use"); NRC Emotion
       Intensity Lexicon (Mohammad, 2018; non-commercial research use)

``data/live_cache_eval/`` holds the Google Cloud Vision answers for datasets A and C, so the web
lookup's evaluation replays without a key. They are web addresses, titles and dates of public pages.


Licences
========

The code is released under the MIT licence (``LICENSE``). The datasets keep their own licences,
listed above. The models keep theirs: CLIP and Whisper (OpenAI, MIT), BLIP (Salesforce,
BSD-3-Clause), spaCy ``en_core_web_md`` (MIT) and Phi-3-mini (Microsoft, MIT).


Known limits
============

These come from the final evaluation; ``results/report_numbers.csv`` has every figure with its
95% interval, and the file and step it comes from.

* **Caption vs picture** compares the kind of scene only. On 165 VERITE pairs it had never seen, it
  flagged 3 of 48 truthful captions and caught 26 of 69 captions taken from other stories, but only
  7 of 48 captions that fit the scene and get a fact wrong. On video it flagged 5 of 34 clips with
  their own caption and 33 of 34 with a caption from another clip.
* **Recycled context** finds a copy only if it is in the offline index or, with the web search,
  on a page Google has seen: the web search found 17 of the 40 dataset A photos online, and could
  date 10 of them. Dates found on web pages are often much later than first publication.
* **Speech vs picture** flagged 13 of 14 spoken lines describing a different scene, but only 2 of
  7 lines with a wrong detail (a colour, a number, a place), and 14 of 93 lines that fit. Whisper
  can invent a sentence in a long silence, and a stretch of speech can start or end a little off.
* **Shouting style** is about capitals, exclamation marks and set words. On 99 meme texts it caught
  5 of 50 manipulative ones (5 of its 10 flags were right); calm manipulative wording passes.
* **The language model** is slow on a CPU (a median of 16 s per answer; 9 of 345 answers timed out
  at 30 s), gave the same answer five times out of five on 22 of 30 pairs, and said "yes, this is
  manipulative" to 23 of 49 honest texts.
* **Speed**: on the development laptop (a 4-core i5 from 2018, 8 GB of RAM) an image takes under a
  second once the models are loaded (13 to 17 s for the first, which loads them), or about 21 s
  with the language model on; a 60-second video takes about 45 s in a fresh process and 24 to 28 s
  after that.
* **No user study**: the interface was checked by automated measures (accessibility, target size,
  text size, reading grade, steps and predicted time per task) and by the author's own
  walkthrough, not by testing with users.
