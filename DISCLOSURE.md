# Responsible Disclosure & Safety Stance

REFUSAL-CLIMB is **AI-safety red-team research**. Its purpose is to make models *harder* to
jailbreak by understanding how the refusal boundary can be searched efficiently, and by pairing
every offensive finding with a defense evaluation. This document states the rules the project
operates under.

## 1. No harmful content in this repository

- There are **no harmful prompts, payloads, or real jailbreaks** anywhere in this repo.
- The "thing the model refuses to say" is a deliberately **benign, arbitrary token**
  (the word `BANANA`). All demonstrations elicit only this harmless token.
- The contribution is a **method** — using continuous refusal-strength as a search gradient —
  not an attack. The method is shown on a proxy precisely so it can be published openly.

## 2. Offline, sandbox-only by construction

- The shipped code runs **fully offline**: no API keys, no model downloads, no network, no GPU.
  The only runtime dependency is `numpy`.
- The real white-box path (extracting a refusal direction from activations) is intended for
  **open-weight models you are authorized to inspect**, run in a **sandbox**. It is gated behind
  `# TODO(real):` markers and is not executed by default.
- Frontier / closed models are **only** ever touched via **authorized black-box channels**
  (e.g., a provider's official API under its terms, or a coordinated red-team engagement).
  This repo never ships such access and never sends operational payloads.

## 3. What we release vs. what we withhold

We release:
- the **method** (refusal-strength-guided search),
- the **taxonomy** of discovered classes (named, abstract framings),
- the **harness** (so defenders can reproduce and measure),
- the **defense evaluation** results.

We **withhold** operational payloads and any concrete, working attack strings against real
models. A taxonomy of *classes* (e.g., "fictional-framing") is defensively useful; a
copy-paste exploit is not.

## 4. Coordinated disclosure of transferable classes

If real-model work (the TODO path) surfaces a **novel class that transfers** to a closed
frontier model:
1. Report privately to the affected provider(s) first, with reproduction steps and the
   abstract class description.
2. Allow a reasonable remediation window before any public discussion.
3. Publish the **class and defense**, not a turnkey exploit.

## 5. Pair offense with defense — always

Every discovery run in this project is followed by a **defense-eval** step (paraphrase /
classifier; blocked-rate). The goal is not to maximize attack success but to identify which
classes survive defenses so they can be *closed*. Findings that a defense already blocks are
reported as such.

## 6. Intended audience and use

This is intended for safety researchers, red teams, and model providers. Do not use the
real-model path against systems you are not authorized to test, or to produce harmful content.

---

*Questions about scope or disclosure: contact the author (Vishnu Kosuri) before extending the
real-model path.*
