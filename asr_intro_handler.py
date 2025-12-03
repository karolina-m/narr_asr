import re
import pathlib
from typing import Tuple, Optional, Dict, Any, List

def detect_introduction_end(text: str) -> Optional[int]:
    """
    Detect where the introduction/warm-up ends in a transcript.
    Returns the character position after the introduction (index), or None if not found.

    This function looks for a variety of patterns that indicate the start of the main task.
    If it finds a match, it returns the start of the line containing that match (so we keep
    consistent segmentation).
    """
    if not text:
        return None

    # Patterns that indicate the start of the main experimental task (Polish-focused)
    task_start_patterns = [
        r'\b(?:popatrz|zobacz)\b.*\b(?:trzy|3)\s+koperty?\b',
        r'\b(?:są|jest)\s+w\s+nich\s+różne\s+historyjki\b',
        r'\bwybierz\s+(?:proszę|,\s*proszę)?\s*(?:jedną|jeden)\b.*\b(?:koperta|historyjka|bajka)\b',
        r'\bmam\s+(?:tu|tutaj)\s+trzy\s+koperty?\b',
        r'\b(?:zadajemy|zadanie|zadanie)\b',  # more general "zadanie"
        r'\b(?:instrukcj[ae]|prosz[ea]\s+wybrać|prosz[ea]\s+przeczytać)\b',
        r'\b(?:poznamy|posłuchaj)\b.*\b(?:historyjki|bajki|opowieści)\b',
        # If instructions include "proszę posłuchać" or "posłuchaj proszę"
        r'\bprosz(?:ę|esz)\b.*\bposłuch(?:ać|aj)\b',
    ]

    earliest_match = None
    earliest_pos = len(text) + 1

    for pattern in task_start_patterns:
        for match in re.finditer(pattern, text, flags=re.IGNORECASE | re.DOTALL):
            if match.start() < earliest_pos:
                earliest_pos = match.start()
                earliest_match = match

    if earliest_match:
        # Find start of the line containing the match
        line_start = text.rfind('\n', 0, earliest_match.start())
        if line_start == -1:
            line_start = 0
        else:
            line_start = line_start + 1  # move after newline
        return line_start

    return None


def extract_introduction_from_hyp(hyp_text: str) -> str:
    """
    Extract the introduction section from the hypothesis (AI) transcript using detect_introduction_end.
    If detection fails, use fallback heuristics (first N words/lines or until a likely task-start word).
    Returns the introduction text, or empty string if nothing sensible found.
    """
    if not hyp_text:
        return ""

    intro_end = detect_introduction_end(hyp_text)
    if intro_end:
        intro = hyp_text[:intro_end].strip()
        # sanity check: if intro is very short (< 3 words) treat as not found
        if len(intro.split()) >= 3:
            return intro
        # otherwise fall through to fallback heuristics

    # FALLBACK HEURISTICS:
    # 1) Look for first blank-line separation (common for transcripts)
    parts = re.split(r'\n\s*\n', hyp_text.strip())
    if parts and len(parts[0].split()) >= 3:
        return parts[0].strip()

    # 2) Look until the first occurrence of common "task" keywords; include some context before them
    keywords = ['zadanie', 'wybierz', 'kopert', 'historyjk', 'bajk', 'instrukcj', 'proszę', 'posłuch', 'popatrz', 'zobacz']
    for kw in keywords:
        m = re.search(rf'.{{0,200}}\b{kw}\w*\b', hyp_text, flags=re.IGNORECASE | re.DOTALL)
        if m:
            # take everything before the match (if at least a few words)
            candidate = hyp_text[:m.start()].strip()
            if len(candidate.split()) >= 3:
                return candidate

    # 3) As last resort, take the first 100-200 words (but only if hyp_text is reasonably long)
    words = hyp_text.split()
    if len(words) <= 200 and len(words) >= 10:
        # short transcript: maybe it's only intro
        return hyp_text.strip()
    if len(words) >= 30:
        return " ".join(words[:120]).strip()

    # nothing useful
    return ""


def remove_introduction_from_hyp(hyp_text: str) -> str:
    """
    Remove the introduction section from hypothesis (AI) transcript.
    Return the remaining text. If intro detection fails or removing would leave nearly-empty
    transcript, do not remove anything (safety).
    """
    if not hyp_text:
        return hyp_text

    intro_end = detect_introduction_end(hyp_text)
    if intro_end:
        remaining = hyp_text[intro_end:].strip()
        # Do not remove intro if remaining is extremely short (likely the audio contained only intro)
        if len(remaining.split()) < 5:
            # keep original
            return hyp_text
        return remaining

    # fallback: try to remove the extracted intro by heuristics
    fallback_intro = extract_introduction_from_hyp(hyp_text)
    if fallback_intro:
        remaining = hyp_text[len(fallback_intro):].strip()
        if len(remaining.split()) < 5:
            return hyp_text
        return remaining

    return hyp_text


def find_placeholders_in_text(ref_text: str) -> List[str]:
    """
    Find bracketed placeholders in a reference transcript, allowing for multiple variants.
    Returns a list of matched bracketed placeholders (strings including brackets).
    Example matches:
      [standardowa rozgrzewka]
      [wstęp pomięty przez osobę spisującą]
      [wstep pominięty]
    """
    if not ref_text:
        return []

    # We'll match any bracketed expression that contains likely placeholder tokens.
    # The token group contains Polish words that indicate "intro", "rozgrzewka", "wstęp", "pominięty", etc.
    token_group = r'(rozgrzewka|wst[eę]p|wstep|pomi|pomini|pomini[eę]ty|pomni[eę]ty|pom?tni[eę]ty|pom[ix]e|pom(.*)ty)'
    pattern = rf'\[[^\]]{{0,160}}(?:{token_group})[^\]]{{0,160}}\]'
    matches = re.findall(pattern, ref_text, flags=re.IGNORECASE | re.DOTALL)

    # re.findall with groups returns tuples; instead use finditer to capture full match text
    full_matches = []
    for m in re.finditer(pattern, ref_text, flags=re.IGNORECASE | re.DOTALL):
        full_matches.append(m.group(0).strip())
    return full_matches


def has_placeholder(ref_text: str) -> bool:
    """
    Binary indicator if reference contains any placeholder-like bracketed tokens.
    """
    return len(find_placeholders_in_text(ref_text)) > 0


def replace_placeholders_with_intro(ref_text: str, intro_text: str) -> str:
    """
    Replace all placeholder bracket expressions that look like intro placeholders with the provided intro_text.
    If no intro_text provided, return ref_text unchanged.
    """
    if not intro_text:
        return ref_text

    placeholders = find_placeholders_in_text(ref_text)
    if not placeholders:
        return ref_text

    modified = ref_text
    for ph in placeholders:
        # replace exact placeholder occurrences (case-insensitive)
        # use re.escape to avoid special chars in placeholder
        modified = re.sub(re.escape(ph), intro_text, modified, flags=re.IGNORECASE)
    return modified


def preprocess_transcript_pair(ref_text: str, hyp_text: str, method: str = "replace") -> Tuple[str, str, Dict[str, Any]]:
    """
    Preprocess a pair of reference and hypothesis transcripts to handle introduction mismatch.

    method:
      - "replace": replace placeholder(s) in ref with extracted intro from hyp
      - "remove": remove intro from hyp (when ref contains placeholder)
      - "noop": do nothing

    Returns:
      processed_ref, processed_hyp, metadata dict (keys: has_placeholder (int), placeholders (list),
      intro_extracted (bool), intro_text (str), action_taken (str))
    """
    metadata: Dict[str, Any] = {
        'has_placeholder': 0,
        'placeholders': [],
        'intro_extracted': False,
        'intro_text': '',
        'action_taken': 'none'
    }

    placeholders = find_placeholders_in_text(ref_text)
    if placeholders:
        metadata['has_placeholder'] = 1
        metadata['placeholders'] = placeholders

    processed_ref = ref_text
    processed_hyp = hyp_text

    method = method.lower()

    # If no placeholder and "remove" method -> do nothing
    if not metadata['has_placeholder'] and method == 'remove':
        metadata['action_taken'] = 'no_placeholder_no_action'
        return processed_ref, processed_hyp, metadata

    # Try to extract intro from hyp once (for both replace/remove)
    intro_text = extract_introduction_from_hyp(hyp_text)
    if intro_text:
        metadata['intro_extracted'] = True
        metadata['intro_text'] = intro_text

    if method == "replace":
        if metadata['has_placeholder']:
            if metadata['intro_extracted']:
                processed_ref = replace_placeholders_with_intro(ref_text, intro_text)
                metadata['action_taken'] = 'replaced_placeholder_with_intro'
            else:
                # If we couldn't extract intro, try a fallback: insert a short marker or leave as-is.
                # We'll leave ref unchanged but record that replace was attempted and failed.
                metadata['action_taken'] = 'placeholder_found_but_no_intro_extracted'
        else:
            metadata['action_taken'] = 'no_placeholder_no_action'

    elif method == "remove":
        if metadata['has_placeholder']:
            # Remove intro from hyp only if we can detect an intro AND removing doesn't leave hyp empty
            candidate_removed = remove_introduction_from_hyp(hyp_text)
            if candidate_removed != hyp_text:
                processed_hyp = candidate_removed
                metadata['action_taken'] = 'removed_intro_from_hyp'
            else:
                metadata['action_taken'] = 'remove_attempted_but_kept_original'
        else:
            metadata['action_taken'] = 'no_placeholder_no_action'

    elif method == "noop":
        metadata['action_taken'] = 'noop'

    else:
        raise ValueError(f"Unknown method: {method}. Use 'replace', 'remove' or 'noop'.")

    return processed_ref, processed_hyp, metadata


# Integration with your existing analysis pipeline
def improved_wer_calculation(reference_path, hypothesis_path, method="replace"):
    """
    Calculate WER with introduction handling. Returns a pandas DataFrame with columns:

    - key, ref_file, hyp_file, WER, CER, WIL, MER, WIP
    - has_placeholder (0/1)
    - placeholder_text (joined placeholders or empty)
    - intro_extracted (0/1)
    - intro_length_chars (int)
    - action_taken (text describing what preprocess did)
    """
    import pandas as pd
    import jiwer
    import os

    # Your existing transformation pipeline (kept; you may refine)
    transforms = jiwer.Compose([
        jiwer.ExpandCommonEnglishContractions(),
        jiwer.RemoveEmptyStrings(),
        jiwer.ToLowerCase(),
        jiwer.RemoveMultipleSpaces(),
        jiwer.Strip(),
        jiwer.RemovePunctuation(),
        jiwer.ReduceToListOfListOfWords(),
    ])

    # List all .txt files
    ref_files = [f for f in os.listdir(reference_path) if f.endswith('.txt')]
    hyp_files = [f for f in os.listdir(hypothesis_path) if f.endswith('.txt')]

    def extract_key(filename):
        # Attempt to preserve your original key extraction rule
        match = re.match(r"^[A-Za-z]{2}\d{3,4}_NAp", filename)
        if match:
            return match.group(0)
        # Fallback: filename without extension
        return pathlib.Path(filename).stem

    df_ref = pd.DataFrame({
        'key': [extract_key(f) for f in ref_files],
        'ref_file': ref_files
    })
    df_hyp = pd.DataFrame({
        'key': [extract_key(f) for f in hyp_files],
        'hyp_file': hyp_files
    })

    df = pd.merge(df_ref, df_hyp, on='key')

    results = []
    for _, row in df.iterrows():
        ref_fp = pathlib.Path(reference_path) / row['ref_file']
        hyp_fp = pathlib.Path(hypothesis_path) / row['hyp_file']
        with open(ref_fp, encoding='utf-8') as f:
            ref_text = f.read()
        with open(hyp_fp, encoding='utf-8') as f:
            hyp_text = f.read()

        # IMPORTANT: Preprocess to handle introduction mismatch and gather metadata
        ref_processed, hyp_processed, metadata = preprocess_transcript_pair(
            ref_text, hyp_text, method=method
        )

        # Apply transformations
        # jiwer transforms return nested lists when ReduceToListOfListOfWords is used,
        # so we flatten them similar to prior code
        ref_transformed = " ".join(sum(transforms(ref_processed), []))
        hyp_transformed = " ".join(sum(transforms(hyp_processed), []))

        # compute metrics
        wer_val = jiwer.wer(ref_transformed, hyp_transformed)
        cer_val = jiwer.cer(ref_transformed, hyp_transformed)
        wil_val = jiwer.wil(ref_transformed, hyp_transformed)
        mer_val = jiwer.mer(ref_transformed, hyp_transformed)
        wip_val = jiwer.wip(ref_transformed, hyp_transformed)

        results.append({
            'key': row['key'],
            'ref_file': row['ref_file'],
            'hyp_file': row['hyp_file'],
            'has_placeholder': metadata.get('has_placeholder', 0),
            'placeholder_text': " | ".join(metadata.get('placeholders', [])),
            'intro_extracted': 1 if metadata.get('intro_extracted', False) else 0,
            'intro_length_chars': len(metadata.get('intro_text', "")),
            'action_taken': metadata.get('action_taken', ''),
            'WER': wer_val,
            'CER': cer_val,
            'WIL': wil_val,
            'MER': mer_val,
            'WIP': wip_val
        })

    return pd.DataFrame(results)
