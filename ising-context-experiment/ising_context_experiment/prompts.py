from dataclasses import dataclass
from typing import List


@dataclass
class PromptSpec:
    prompt: str
    expected: str
    competitor: str
    anchors: List[str]


def default_prompt_bank() -> List[PromptSpec]:
    return [
        PromptSpec(
            prompt=(
                "Mira packed her suitcase for the conference. "
                "She put in her laptop, passport, and a printed ticket. "
                "At the airport check-in desk, the agent asked for her"
            ),
            expected=" passport",
            competitor=" umbrella",
            anchors=["airport", "check-in", "passport"],
        ),
        PromptSpec(
            prompt=(
                "The chemistry student wore gloves and goggles. "
                "She measured sodium bicarbonate, added vinegar, and "
                "watched bubbles form. "
                "The gas released in this reaction is"
            ),
            expected=" carbon",
            competitor=" oxygen",
            anchors=["sodium", "vinegar", "reaction"],
        ),
        PromptSpec(
            prompt=(
                "In the story, the detective found muddy footprints, "
                "a broken lock, and an empty jewelry box. "
                "The evidence suggested someone had entered through "
                "the back door and"
            ),
            expected=" stolen",
            competitor=" painted",
            anchors=["footprints", "broken", "jewelry"],
        ),
        PromptSpec(
            prompt=(
                "The teacher wrote 2, 4, 6, 8 on the board and asked "
                "for the next number. "
                "The student recognized the pattern and answered"
            ),
            expected=" 10",
            competitor=" 9",
            anchors=["2", "4", "8"],
        ),
        PromptSpec(
            prompt=(
                "After three days of clouds, the forecast finally "
                "showed a bright sun icon. "
                "People planned picnics because the weather would be"
            ),
            expected=" sunny",
            competitor=" snowy",
            anchors=["forecast", "sun", "picnics"],
        ),
        PromptSpec(
            prompt=(
                "The recipe said to preheat the oven, whisk eggs with "
                "sugar, and fold in flour. "
                "After baking for 30 minutes, the cake should be"
            ),
            expected=" ready",
            competitor=" frozen",
            anchors=["oven", "baking", "cake"],
        ),
        PromptSpec(
            prompt=(
                "The nurse checked the patient's pulse, blood pressure, "
                "and temperature. "
                "Because the temperature was high, she suspected a"
            ),
            expected=" fever",
            competitor=" fracture",
            anchors=["pulse", "temperature", "high"],
        ),
        PromptSpec(
            prompt=(
                "A gardener planted seeds in spring, watered them daily, "
                "and placed them in sunlight. "
                "After a few weeks, tiny green leaves began to"
            ),
            expected=" grow",
            competitor=" melt",
            anchors=["seeds", "watered", "sunlight"],
        ),
    ]
