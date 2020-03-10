import json
import logging
from typing import Dict, List

from overrides import overrides
import numpy

from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import (
    Field,
    ArrayField,
    ListField,
    IndexField,
    TextField,
    MetadataField,
)
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from allennlp.data.tokenizers import Token
from allennlp.data.tokenizers.whitespace_tokenizer import WhitespaceTokenizer


logger = logging.getLogger(__name__)


@DatasetReader.register("cnn_dm")
class CnnDmReader(DatasetReader):
    """
    Reads the CNN / DailyMail datasets into a ``Dataset`` containing ``Instances`` with four fields:
    ``question`` (a ``TextField``), ``context`` (another ``TextField``), ``label`` (a ``LabelField``),
    and ``label_mask`` (a ``ArrayField``).

    This reads preprocessed JSON versions of the original CNN / DailyMail datasets. The original datasets
    are tarballs with millions of documents in a rather inconvenient and space-consuming format.

    The dataset consists of space-delimited tokens, so we use a WhitespaceTokenizer.

    Parameters
    ----------
    token_indexers : ``Dict[str, TokenIndexer]``, optional
        We similarly use this for both the question and the context.  See :class:`TokenIndexer`.
        Default is ``{"tokens": SingleIdTokenIndexer()}``.
    relabel_entities : ``bool``, optional (default=True)
        If this is true, relabel the entities to start from zero and increment by one. Change the
        label accordingly as well.
    lazy : ``bool``, optional (default=False)
        If this is true, ``instances()`` will return an object whose ``__iter__`` method
        reloads the dataset each time it's called. Otherwise, ``instances()`` returns a list.
    """

    def __init__(
        self,
        token_indexers: Dict[str, TokenIndexer] = None,
        relabel_entities: bool = True,
        lazy: bool = False
    ) -> None:
        super().__init__(lazy)
        self._tokenizer = WhitespaceTokenizer()
        self._relabel_entities = relabel_entities
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}

    @overrides
    def _read(self, file_path: str):
        # if `file_path` is a URL, redirect to the cache
        file_path = cached_path(file_path)

        logger.info("Reading file at %s", file_path)
        with open(file_path) as dataset_file:
            for instance_line in dataset_file:
                instance = json.loads(instance_line)
                yield self.text_to_instance(
                    question_text=instance["question"],
                    context_text=instance["context"],
                    answer_text=instance["answer"])

    @overrides
    def text_to_instance(
        self,  # type: ignore
        question_text: str,
        context_text: str,
        answer_text: str,
        question_tokens: List[Token] = None,
        context_tokens: List[Token] = None,
    ) -> Instance:
        fields: Dict[str, Field] = {}
        if not question_tokens:
            question_tokens = self._tokenizer.tokenize(question_text)
        if not context_tokens:
            context_tokens = self._tokenizer.tokenize(context_text)
        if self._relabel_entities:
            # Relabel entities to start from @entity0 and count up for each unique entity.
            new_entity_ids = {}
            for word in context_tokens + question_tokens:
                if word.text.startswith('@entity') and word.text not in new_entity_ids:
                    new_entity_ids[word.text] = '@entity' + str(len(new_entity_ids))
            question_tokens = [Token(new_entity_ids.get(token.text, token.text)) for token in question_tokens]
            context_tokens = [Token(new_entity_ids.get(token.text, token.text)) for token in context_tokens]
            if answer_text:
                answer_text = new_entity_ids[answer_text]
        context_field = TextField(context_tokens, self._token_indexers)
        fields["context"] = context_field
        fields["question"] = TextField(question_tokens, self._token_indexers)
        fields["context_entity_mask"] = ArrayField(numpy.array([1 if tok.text.startswith("@entity") else 0 for
                                                                tok in context_tokens]))
        metadata = {
            "original_context": context_text,
            "question_tokens": [token.text for token in question_tokens],
            "context_tokens": [token.text for token in context_tokens]
        }
        if self._relabel_entities:
            metadata["new_entity_ids"] = new_entity_ids
        if answer_text:
            # For each occurrence of answer_text in the context, make an IndexField
            context_label_index_fields = [IndexField(token_index, context_field) for
                                          token_index, token in enumerate(context_tokens) if
                                          token.text == answer_text]
            fields["answer_as_passage_indices"] = ListField(context_label_index_fields)
            metadata["answer_text"] = answer_text
        fields["metadata"] = MetadataField(metadata)
        return Instance(fields)