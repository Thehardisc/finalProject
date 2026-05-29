from collections import Counter

import emoji as emoji_lib
import torch
import torch.nn.functional as F

from shared.constants import EMOTION_LABELS


EMOTION_ANCHORS = {
    'admiration':    'That is so impressive and admirable',
    'amusement':     'This is so funny and entertaining',
    'anger':         'I am so angry and furious right now',
    'annoyance':     'This is really annoying and irritating',
    'approval':      'I agree and approve of this, great job',
    'caring':        'I care about you and want to help',
    'confusion':     'I am confused and do not understand',
    'curiosity':     'I am curious and want to know more',
    'desire':        'I really want this so badly',
    'disappointment':'I am so disappointed and let down',
    'disapproval':   'I disagree and disapprove of this',
    'disgust':       'This is disgusting and repulsive',
    'embarrassment': 'I feel so embarrassed and ashamed',
    'excitement':    'I am so excited and thrilled',
    'fear':          'I am scared and frightened',
    'gratitude':     'Thank you so much, I am very grateful',
    'grief':         'I am devastated and grieving deeply',
    'joy':           'I feel so happy and joyful',
    'love':          'I love you with all my heart, hearts full of love and affection',
    'nervousness':   'I am nervous and anxious',
    'optimism':      'I feel hopeful and optimistic about this',
    'pride':         'I am so proud of this achievement',
    'realization':   'I just realized and understood something important',
    'relief':        'I feel so relieved, thank goodness',
    'remorse':       'I feel terrible and regret this deeply',
    'sadness':       'I feel so sad and devastated, I am crying tears of sadness',
    'surprise':      'I am completely shocked and surprised',
    'neutral':       'This is a plain neutral statement with no emotion',
}

# 5.0 gives clearer signal than 3.0 without over-concentrating on one label.
SOFTMAX_TEMPERATURE = 5.0


class EmojiSemanticScorer:
    """
    Score emojis by cosine similarity to 28 anchor-phrase embeddings.

    Construct once per process/cycle — the anchor pre-compute embeds 28
    phrases through the encoder, then per-emoji lookup is a single encode
    plus a tiny cosine. Reusing the same instance amortizes the setup.
    """

    def __init__(self, hf_pipeline):
        """
        Args:
            hf_pipeline: A transformers text-classification pipeline. We
                use its tokenizer and the base encoder of its model
                (handles both BERT and RoBERTa heads).
        """
        self._tokenizer = hf_pipeline.tokenizer
        self._model = hf_pipeline.model
        self._anchors = {
            label: self._encode(EMOTION_ANCHORS.get(label, label))
            for label in EMOTION_LABELS
        }

    def _encode(self, text: str) -> torch.Tensor:
        inputs = self._tokenizer(text, return_tensors="pt",
                                 truncation=True, max_length=64)
        encoder = (getattr(self._model, 'bert', None)
                   or getattr(self._model, 'roberta', None)
                   or self._model.base_model)
        with torch.no_grad():
            outputs = encoder(**inputs)
        hidden = outputs.last_hidden_state
        mask = inputs['attention_mask'].unsqueeze(-1).float()
        return (hidden * mask).sum(dim=1) / mask.sum(dim=1)

    def analyze(self, text: str) -> dict:
        """
        Returns a 28-label probability dict. All zeros if no emojis found
        (callers can pass the result directly into build_fv / build_feature_vector).
        """
        zero = {label: 0.0 for label in EMOTION_LABELS}
        occurrences = emoji_lib.emoji_list(text)
        if not occurrences:
            return zero

        counts = Counter(e["emoji"] for e in occurrences)
        total = sum(counts.values())
        weighted = dict(zero)

        for ch, cnt in counts.items():
            desc = emoji_lib.demojize(ch).strip(":").replace("_", " ")
            if not desc:
                continue
            try:
                emoji_emb = self._encode(desc)
                sims = torch.tensor([
                    F.cosine_similarity(emoji_emb, self._anchors[label]).item()
                    for label in EMOTION_LABELS
                ])
                probs = F.softmax(sims * SOFTMAX_TEMPERATURE, dim=0).tolist()
            except Exception:
                continue

            w = cnt / total
            for label, p in zip(EMOTION_LABELS, probs):
                weighted[label] += p * w

        return weighted
