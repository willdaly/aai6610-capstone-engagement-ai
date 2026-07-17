"""Generate the mini-corpus fixture used by the assistant tests.

The content here is entirely INVENTED: a fictional "Larkspur syndrome" and a fictional
"Larkspur Family Foundation" on an example.org domain. None of it is real Sturge-Weber
Foundation text. That is deliberate and required by the plan: no real corpus text may
appear in any test fixture. The pages only mimic the structure (headings, paragraphs,
a contact page, an educational page with medical-adjacent content) so the chunker,
index, retrieval, refusal, escalation, and CLI paths can be exercised offline.

Run ``python assistant/tests/fixtures/build_fixture.py`` to regenerate the JSON page
files under ``mini_corpus/pages/``. The committed JSON is the fixture; this builder just
keeps it easy to edit without hand-maintaining word counts.
"""

from __future__ import annotations

import json
from pathlib import Path

FIXTURE_DIR = Path(__file__).resolve().parent / "mini_corpus" / "pages"
BASE = "https://larkspur.example.org"


def h(text: str) -> dict:
    return {"kind": "heading", "text": text}


def p(text: str) -> dict:
    return {"kind": "text", "text": " ".join(text.split())}


PAGES = [
    {
        "url": f"{BASE}/who-we-are/mission.html",
        "title": "Our Mission - Larkspur Family Foundation",
        "blocks": [
            h("Our Mission"),
            p("""The Larkspur Family Foundation supports families affected by Larkspur
               syndrome, a rare condition invented for this fixture. We connect families
               with education, community, and the clinics that understand the condition.
               We are a small nonprofit run mostly by volunteers and parents who have
               walked this road before. Our work centers on the families we serve, and
               everything we publish is meant to help a caregiver find their footing."""),
            h("What We Do"),
            p("""We publish plain-language education about Larkspur syndrome, maintain a
               directory of clinics in our care network, host an annual family gathering,
               and offer a small volunteer-run helpline. We do not provide medical care
               and we do not give medical advice. Instead we point families toward the
               clinicians and resources that can help them make decisions with their own
               care team. Membership is free and open to any affected family."""),
        ],
    },
    {
        "url": f"{BASE}/education/understanding-larkspur.html",
        "title": "Understanding Larkspur Syndrome",
        "blocks": [
            h("What is Larkspur syndrome?"),
            p("""Larkspur syndrome is a fictional congenital condition used only to test
               this assistant. In this made-up description it involves a blue skin marking
               present at birth and a chance of eye and neurological differences. It is
               not inherited in any predictable pattern and does not run in families in
               this invented account. The rest of this page describes made-up educational
               material a foundation might publish about the condition."""),
            h("Common features"),
            p("""In this fixture the described features include a blue birthmark, a raised
               chance of increased eye pressure, and a possibility of seizures in some
               children. The severity varies widely from one child to the next. Some
               children have only the skin marking and no other findings at all, while
               others have eye or neurological involvement that needs regular follow-up
               with specialists who know the condition."""),
            h("Watching the eyes"),
            p("""Increased eye pressure can develop early, so the invented guidance here
               is that babies with a marking near the eye should be seen by an eye doctor.
               This is general education, not a recommendation for any specific child. Any
               decision about a specific child's eyes belongs to that family and their own
               eye doctor, who can examine the child and advise them directly."""),
            h("Seizures"),
            p("""When seizures occur in this fictional condition, families in the story are
               told to work with a neurologist. The foundation's role is to help families
               find such specialists and to explain terms in plain language, never to
               recommend a particular medication, dose, or treatment plan for a child.
               Treatment choices are made by a family together with their clinicians."""),
        ],
    },
    {
        "url": f"{BASE}/for-patients/care-network.html",
        "title": "Larkspur Clinical Care Network",
        "blocks": [
            h("The Care Network"),
            p("""The Larkspur Clinical Care Network is a fictional list of clinics that
               have said they are comfortable seeing children with Larkspur syndrome. The
               network exists so families do not have to explain a rare condition from
               scratch at every visit. Clinics in the network span eye care, neurology,
               and general pediatrics in this invented directory."""),
            h("Finding a clinic"),
            p("""To find a clinic in this made-up network, families are directed to the
               care network page and encouraged to call the foundation's helpline if they
               need help choosing. The foundation does not refer to a specific clinician
               or endorse one clinic over another; it simply lists the clinics that have
               joined the network so families can start a conversation with their own
               care team about where to go."""),
        ],
    },
    {
        "url": f"{BASE}/who-we-are/contact.html",
        "title": "Contact the Larkspur Family Foundation",
        "blocks": [
            h("Contact Us"),
            p("""You can reach the Larkspur Family Foundation by email at
               hello@larkspur.example.org or by calling the volunteer helpline at
               555-0100 during weekday afternoons. This contact information is invented
               for the fixture. A real person, usually a parent volunteer, answers the
               helpline and can help you find pages on this site or connect you with a
               clinic in the care network."""),
            h("Where we are"),
            p("""The foundation is a distributed group of volunteers rather than a single
               office. The best way to reach a human is the helpline or the email address
               above. We aim to reply within a few business days, and we never ask for
               medical records or personal health details over email."""),
        ],
    },
    {
        "url": f"{BASE}/events/family-gathering.html",
        "title": "Annual Larkspur Family Gathering",
        "blocks": [
            h("Annual Family Gathering"),
            p("""The annual Larkspur Family Gathering is a fictional event where families
               meet, hear from clinicians in the care network, and spend time with others
               who understand the condition. In this invented description the gathering
               happens each spring and moves to a different city each year. Registration
               is free for families and opens a few months ahead of the event."""),
            h("What to expect"),
            p("""Sessions in the story include plain-language talks, a children's program,
               and time to ask questions of volunteer parents. The foundation covers some
               travel costs for families who need help attending. Details and dates are
               posted on this events page when they are set, and the helpline can answer
               questions about attending."""),
        ],
    },
]


def main() -> None:
    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    for page in PAGES:
        extracted = "\n".join(b["text"] for b in page["blocks"])
        slug = page["url"].rsplit("/", 1)[-1].replace(".html", "")
        payload = {
            "url": page["url"],
            "title": page["title"],
            "http_status": 200,
            "word_count": len(extracted.split()),
            "blocks": page["blocks"],
        }
        (FIXTURE_DIR / f"{slug}.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        print(f"wrote {slug}.json ({payload['word_count']} words)")


if __name__ == "__main__":
    main()
