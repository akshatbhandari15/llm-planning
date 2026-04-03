"""
Improved prompt datasets for VOMC-QKV planning detection.

The key insight: planning is detectable when early context CONSTRAINS
future tokens in a non-trivial way. We need prompts where:

1. Information introduced early becomes critical several tokens later
2. The correct continuation is unambiguous given full context but
   ambiguous without it
3. There's a "delayed payoff" — the model must hold state across
   multiple generation steps

Three categories of prompts are provided, each targeting a different
aspect of planning behavior.
"""

# ─────────────────────────────────────────────────────────────────────
# CATEGORY A: GARDEN-PATH / DISAMBIGUATION PROMPTS
#
# These prompts set up an ambiguity early on that gets resolved by
# later context. The model must "plan" in the sense that the V state
# at early positions should already shift once disambiguating context
# arrives — and future tokens should become predictable.
# ─────────────────────────────────────────────────────────────────────

GARDEN_PATH_PROMPTS = [
    {
        "prompt": "The old man the boats while the young women watch from the shore and occasionally",
        "note": "Garden path: 'old' and 'man' are adjective+verb, not adj+noun",
        "expected_planning": "the V state should shift at 'the boats' when 'man' is reinterpreted as a verb",
    },
    {
        "prompt": "The horse raced past the barn fell down and the veterinarian rushed over to examine",
        "note": "Reduced relative clause: 'raced past the barn' modifies 'horse'",
        "expected_planning": "planning should spike at 'fell' which reframes the whole sentence",
    },
    {
        "prompt": "Because he always jogs a mile seems like a very short distance to",
        "note": "Temporary ambiguity: 'jogs a mile' vs 'jogs' (intransitive) + 'a mile seems'",
        "expected_planning": "resolution at 'seems' should restructure future planning",
    },
    {
        "prompt": "The cotton clothing is made of grows in fields across the southern states of",
        "note": "'cotton' is the subject, not modifier of 'clothing'",
        "expected_planning": "'grows' forces reanalysis; future tokens become geographically constrained",
    },
    {
        "prompt": "While the musician played the piano was being tuned by a technician in the back of the",
        "note": "'played' is intransitive here; 'the piano' starts a new clause",
        "expected_planning": "disambiguation at 'was being tuned' should shift planning",
    },
]

# ─────────────────────────────────────────────────────────────────────
# CATEGORY B: LONG-RANGE DEPENDENCY PROMPTS
#
# These prompts introduce a constraint early that must be honored
# many tokens later. The model needs to "remember" and "plan" to
# maintain agreement, complete a pattern, or fulfill a setup.
# ─────────────────────────────────────────────────────────────────────

LONG_RANGE_PROMPTS = [
    {
        "prompt": "The three brothers, who had been separated since childhood, finally reunited at their mother's funeral. The oldest spoke first, then the middle one, and finally the youngest said",
        "note": "Must maintain the 'three brothers' count and birth order across the whole passage",
        "expected_planning": "MI should be high between 'three' and future tokens about 'youngest'",
    },
    {
        "prompt": "She opened the letter, which read: Dear Dr. Thompson, I am writing to inform you that your application for the position of",
        "note": "The 'Dr.' title constrains the job to be professional/academic",
        "expected_planning": "early tokens (Dr., Thompson) should predict professional role tokens",
    },
    {
        "prompt": "First, preheat the oven to 350 degrees. Second, mix the dry ingredients. Third, add the wet ingredients. Fourth, pour into the pan. Fifth,",
        "note": "Numbered list pattern forces continuation with ordinal + cooking step",
        "expected_planning": "strong sequential planning; current state should predict 'bake' or similar",
    },
    {
        "prompt": "In 1969, Neil Armstrong became the first person to walk on the moon. Exactly fifty years later, in",
        "note": "Arithmetic constraint: 1969 + 50 = 2019",
        "expected_planning": "the state at '1969' and 'fifty' should jointly predict '2019'",
    },
    {
        "prompt": "The defendant was charged with robbery on Monday, arraigned on Tuesday, and the trial was scheduled for Wednesday. On Thursday, the jury announced their",
        "note": "Temporal sequence constrains the outcome to be a verdict",
        "expected_planning": "the legal context should make 'verdict' highly predictable from early state",
    },
    {
        "prompt": "To make a peanut butter and jelly sandwich, you need bread, peanut butter, and jelly. First, take two slices of bread. Then, spread peanut butter on one slice and",
        "note": "Strong procedural constraint: next step must involve jelly",
        "expected_planning": "'peanut butter AND jelly' early on should predict jelly-related tokens",
    },
    {
        "prompt": "The patient presented with a fever of 104 degrees, severe headache, and a stiff neck. The doctor immediately suspected bacterial",
        "note": "Medical symptom triad (fever + headache + stiff neck) strongly predicts meningitis",
        "expected_planning": "early symptoms should collectively predict 'meningitis'",
    },
    {
        "prompt": "If x equals 5 and y equals 10, then x plus y equals",
        "note": "Arithmetic: clear deterministic answer",
        "expected_planning": "state should encode both values and predict '15'",
    },
    {
        "prompt": "The rhyme scheme of this poem is ABAB. The first line ends with 'day', the second with 'night', the third with 'way', and the fourth must end with a word that rhymes with",
        "note": "Pattern forces rhyme with 'night'",
        "expected_planning": "ABAB pattern + 'night' should strongly predict rhyming words",
    },
    {
        "prompt": "Knock knock. Who's there? Banana. Banana who? Knock knock. Who's there? Banana. Banana who? Knock knock. Who's there? Orange. Orange who? Orange you glad I didn't say",
        "note": "Classic joke pattern: repetition then punchline",
        "expected_planning": "the repeated 'banana' pattern should predict 'banana' at the end",
    },
]

# ─────────────────────────────────────────────────────────────────────
# CATEGORY C: CONTEXTUAL RAMP-UP PROMPTS
#
# These are single long passages where we progressively reveal more
# context. Unlike the factual prompts, these have a "ramp" structure
# where each additional sentence significantly constrains the
# continuation. Designed specifically for the context sweep (Phase 2).
# ─────────────────────────────────────────────────────────────────────

CONTEXTUAL_RAMP_PROMPTS = [
    {
        "full_text": "The detective examined the crime scene carefully. Blood stains covered the carpet near the fireplace. A broken window suggested forced entry from outside. The victim's wallet was missing but expensive jewelry remained untouched. This pattern clearly indicated that the motive was",
        "expected": " not",
        "note": "Progressive evidence narrows motive; 'not robbery' becomes clear only with full context",
    },
    {
        "full_text": "Water is composed of hydrogen and oxygen atoms. Each molecule contains exactly two hydrogen atoms bonded to one oxygen atom. The chemical formula that represents this molecular composition is",
        "expected": " H",
        "note": "Scientific facts build toward formula H2O",
    },
    {
        "full_text": "The concert hall was packed with thousands of fans. The drummer counted off the beat. The bassist laid down a groove. The guitarist played a riff. Then the lead singer grabbed the microphone and began to",
        "expected": " sing",
        "note": "Band setup makes 'sing' highly predictable but only with full context",
    },
    {
        "full_text": "In chess, the king can move one square in any direction. The queen can move any number of squares in any direction. The rook can move any number of squares but only horizontally or vertically. The bishop can move any number of squares but only",
        "expected": " diagonal",
        "note": "Pattern of chess piece descriptions constrains bishop's movement",
    },
    {
        "full_text": "The recipe called for flour, sugar, eggs, and butter. She mixed them into a smooth batter, poured it into a greased pan, and placed it in the preheated oven. After thirty minutes of baking, she pulled out a perfectly golden",
        "expected": " cake",
        "note": "Ingredients + process -> cake/bread; full context needed",
    },
    {
        "full_text": "Japan is an island nation in East Asia located in the Pacific Ocean. Its capital city, which is also its largest metropolis and the most populous city in the world, is",
        "expected": " Tokyo",
        "note": "Geographic context builds to capital city",
    },
    {
        "full_text": "The programmer stared at the error message on her screen. The function expected an integer but received a string. She needed to convert the input data type. In Python, she would use the built-in function called",
        "expected": " int",
        "note": "Technical context narrows to specific function",
    },
    {
        "full_text": "During photosynthesis, plants absorb carbon dioxide from the air and water from the soil. Using energy from sunlight, they convert these raw materials into glucose and release a gas that animals need to breathe. This gas is",
        "expected": " oxygen",
        "note": "Biology process description builds to specific product",
    },
]


# ─────────────────────────────────────────────────────────────────────
# IMPROVED NARRATIVE PROMPTS FOR TRAJECTORY GENERATION
#
# These are longer, more structured than the originals, with built-in
# constraints that should make planning detectable.
# ─────────────────────────────────────────────────────────────────────

STRUCTURED_NARRATIVE_PROMPTS = [
    "The professor began the lecture by writing three equations on the board. The first equation described",
    "After examining the patient, the doctor reviewed the blood test results, checked the X-ray, and then told the patient that the diagnosis was",
    "The recipe requires exactly five steps. Step one is to gather ingredients. Step two is to",
    "The witness testified that on the night of December 3rd, she saw the defendant enter the building at precisely",
    "According to the periodic table, the element with atomic number 79, commonly known as gold, has the chemical symbol",
    "The spacecraft launched from Cape Canaveral on a mission to Mars. After traveling for seven months through space, the rover finally landed on the surface and began to",
    "In the game of basketball, each team has five players on the court. The point guard typically handles the ball and calls the plays. The center usually positions near the basket and tries to",
    "The treaty was signed by three nations: France, Germany, and the United Kingdom. The French representative signed first, followed by the German delegate, and finally the British ambassador",
    "The algorithm works in three stages: first it sorts the input data, then it removes duplicates, and finally it",
    "The four seasons of the year are spring, summer, autumn, and winter. After the cold winter months, the snow melts and flowers begin to bloom during",
    "My grocery list has five items: milk, eggs, bread, apples, and cheese. At the store, I found the milk and eggs right away. Next I need to find the",
    "The solar system has eight planets. Starting from the sun: Mercury, Venus, Earth, Mars, Jupiter, Saturn, Uranus, and",
]


# ─────────────────────────────────────────────────────────────────────
# PARAMETER RECOMMENDATIONS
# ─────────────────────────────────────────────────────────────────────

RECOMMENDED_PARAMS = {
    "generation_length": 40,        # Was 20; longer trajectories give more MI samples
    "n_trajectories": 50,           # Was 30; more data for statistical power
    "max_lookahead": 10,            # Keep at 10
    "n_permutations": 200,          # Was 100; tighter p-values
    "n_clusters": 32,               # Keep moderate
    "max_order": 6,                 # Keep at 6
    "temperatures": [0.3, 0.6, 0.9, 1.2],  # Wider range, including Kshitig's hallucination-inducing temps
    "context_lengths": [4, 8, 16, 32, 64, 128],  # Drop 1 and 2 (too short to plan)
    "target_layers": "0,3,6,9,11",  # Sample across depth instead of all layers
}