import torch
from torch import nn


class TokenLearnedEncoding(nn.Module):
    """
    Learned additive img/word/action token encoding implemented on top of nn.Embedding
    """

    def __init__(self, d_model, vocab_size=3, init_range=0.1):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model)
        self.emb.weight.data.uniform_(-init_range, init_range)

    def forward(self, lang, frames, actions):
        token_lang = torch.ones(lang.shape[:2], device=lang.device, dtype=
            torch.long) * 0
        token_lang_emb = self.emb(token_lang)
        lang += token_lang_emb
        token_frames = torch.ones(frames.shape[:2], device=frames.device,
            dtype=torch.long) * 1
        token_frames_emb = self.emb(token_frames)
        frames += token_frames_emb
        token_actions = torch.ones(actions.shape[:2], device=actions.device,
            dtype=torch.long) * 2
        token_actions_emb = self.emb(token_actions)
        actions += token_actions_emb
        return lang, frames, actions


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand(
        [4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'d_model': 4}]
