import torch


def get_args():
    parser = argparse.ArgumentParser()
    group = parser.add_argument_group(title='input data')
    group.add_argument('--input', type=str, required=True, help=
        'Path to input JSON')
    group.add_argument('--json-keys', nargs='+', default=['text'], help=
        'space separate listed of keys to extract from json')
    group.add_argument('--split-sentences', action='store_true', help=
        'Split documents into sentences.')
    group.add_argument('--keep-newlines', action='store_true', help=
        'Keep newlines between sentences when splitting.')
    group = parser.add_argument_group(title='tokenizer')
    group.add_argument('--tokenizer-type', type=str, required=True, choices
        =['BertWordPieceLowerCase', 'BertWordPieceCase', 'GPT2BPETokenizer'
        ], help='What type of tokenizer to use.')
    group.add_argument('--vocab-file', type=str, default=None, help=
        'Path to the vocab file')
    group.add_argument('--merge-file', type=str, default=None, help=
        'Path to the BPE merge file (if necessary).')
    group.add_argument('--append-eod', action='store_true', help=
        'Append an <eod> token to the end of a document.')
    group = parser.add_argument_group(title='output data')
    group.add_argument('--output-prefix', type=str, required=True, help=
        'Path to binary output file without suffix')
    group.add_argument('--dataset-impl', type=str, default='mmap', choices=
        ['lazy', 'cached', 'mmap'])
    group = parser.add_argument_group(title='runtime')
    group.add_argument('--workers', type=int, default=1, help=
        'Number of worker processes to launch')
    group.add_argument('--log-interval', type=int, default=100, help=
        'Interval between progress updates')
    args = parser.parse_args()
    args.keep_empty = False
    if args.tokenizer_type.lower().startswith('bert'):
        if not args.split_sentences:
            None
    args.rank = 0
    args.make_vocab_size_divisible_by = 128
    args.tensor_model_parallel_size = 1
    args.vocab_extra_ids = 0
    return args


def init_method_normal(sigma):
    """Init method based on N(0, sigma)."""

    def init_(tensor):
        return torch.nn.init.normal_(tensor, mean=0.0, std=sigma)
    return init_


class MegatronModule(torch.nn.Module):
    """Megatron specific extensions of torch Module with support
    for pipelining."""

    def __init__(self, share_word_embeddings=True):
        super(MegatronModule, self).__init__()
        self.share_word_embeddings = share_word_embeddings

    def state_dict_for_save_checkpoint(self, destination=None, prefix='',
        keep_vars=False):
        """Use this function to override the state dict for
        saving checkpoints."""
        return self.state_dict(destination, prefix, keep_vars)

    def word_embeddings_weight(self):
        if mpu.is_pipeline_first_stage(ignore_virtual=True):
            return self.language_model.embedding.word_embeddings.weight
        if mpu.is_pipeline_last_stage(ignore_virtual=True):
            if not self.share_word_embeddings:
                raise Exception(
                    'word_embeddings_weight() called for last stage, but share_word_embeddings is false'
                    )
            return self.word_embeddings.weight
        raise Exception(
            'word_embeddings_weight() should be called for first and last stage only'
            )

    def initialize_word_embeddings(self, init_method_normal):
        args = get_args()
        if not self.share_word_embeddings:
            raise Exception(
                'initialize_word_embeddings() was called but share_word_embeddings is false'
                )
        if args.pipeline_model_parallel_size == 1:
            return
        if mpu.is_pipeline_last_stage():
            assert not mpu.is_pipeline_first_stage()
            self._word_embeddings_for_head_key = 'word_embeddings_for_head'
            self.word_embeddings = mpu.VocabParallelEmbedding(args.
                padded_vocab_size, args.hidden_size, init_method=
                init_method_normal(args.init_method_std))
            self.word_embeddings.weight.data.fill_(0)
            self.word_embeddings.weight.shared = True
        if torch.distributed.is_initialized():
            if mpu.is_pipeline_first_stage() or mpu.is_pipeline_last_stage():
                torch.distributed.all_reduce(self.word_embeddings_weight().
                    data, group=mpu.get_embedding_group())
        else:
            None


class VitMlpHead(MegatronModule):
    """Pooler layer.

    Pool hidden states of a specific token (for example start of the
    sequence) and add a linear transformation followed by a tanh.

    Arguments:
        hidden_size: hidden size
        init_method: weight initialization method for the linear layer.
            bias is set to zero.
    """

    def __init__(self, hidden_size, num_classes):
        super(VitMlpHead, self).__init__()
        self.dense_in = torch.nn.Linear(hidden_size, hidden_size)
        self.dense_out = torch.nn.Linear(hidden_size, num_classes)
        torch.nn.init.constant_(self.dense_out.bias, -10)

    def forward(self, hidden_states, sequence_index=0):
        x = hidden_states[:, sequence_index, :]
        x = self.dense_in(x)
        x = torch.tanh(x)
        x = self.dense_out(x)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'hidden_size': 4, 'num_classes': 4}]
