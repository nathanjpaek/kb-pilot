import torch
import torch.utils.data


class SimilarityMatrix(torch.nn.Module):

    def __init__(self, padding=0):
        super().__init__()
        self.padding = padding

    def forward(self, query_embed, doc_embed, query_tok, doc_tok):
        simmat = []
        assert type(query_embed) == type(doc_embed)
        if not isinstance(query_embed, list):
            query_embed, doc_embed = [query_embed], [doc_embed]
        for a_emb, b_emb in zip(query_embed, doc_embed):
            BAT, A, B = a_emb.shape[0], a_emb.shape[1], b_emb.shape[1]
            if a_emb is None and b_emb is None:
                sim = query_tok.reshape(BAT, A, 1).expand(BAT, A, B
                    ) == doc_tok.reshape(BAT, 1, B).expand(BAT, A, B).float()
            else:
                a_denom = a_emb.norm(p=2, dim=2).reshape(BAT, A, 1).expand(BAT,
                    A, B) + 1e-09
                b_denom = b_emb.norm(p=2, dim=2).reshape(BAT, 1, B).expand(BAT,
                    A, B) + 1e-09
                perm = b_emb.permute(0, 2, 1)
                sim = a_emb.bmm(perm) / (a_denom * b_denom)
            nul = torch.zeros_like(sim)
            sim = torch.where(query_tok.reshape(BAT, A, 1).expand(BAT, A, B
                ) == self.padding, nul, sim)
            sim = torch.where(doc_tok.reshape(BAT, 1, B).expand(BAT, A, B) ==
                self.padding, nul, sim)
            simmat.append(sim)
        return torch.stack(simmat, dim=1)


def get_inputs():
    return [torch.rand([4, 4, 4]), torch.rand([4, 4, 4]), torch.rand([4, 4]
        ), torch.rand([4, 4])]


def get_init_inputs():
    return [[], {}]
