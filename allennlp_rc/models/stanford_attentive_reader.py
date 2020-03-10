import logging
from typing import Any, Dict, List, Optional

import torch
from torch.nn.functional import nll_loss

from allennlp.common.checks import check_dimensions_match
from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.modules import (
    Seq2SeqEncoder,
    TimeDistributed,
    TextFieldEmbedder)
from allennlp.modules.matrix_attention import MatrixAttention
from allennlp.nn import util, InitializerApplicator, RegularizerApplicator
from allennlp_rc.eval import SquadEmAndF1

logger = logging.getLogger(__name__)


@Model.register("stanford_attentive_reader")
class StanfordAttentiveReader(Model):
    """This class implements the `Stanford Attentive Reader model
    <https://arxiv.org/abs/1606.02858>`_ for answering questions
    from the CNN / DailyMail RC dataset (ACL 2016).

    The basic layout is pretty simple: encode words as a combination of word
    embeddings, pass the word representations through a bi-LSTM/GRU, use a
    matrix of attentions to put question information into the context word
    representations (this is the only part that is at all non-standard), pass
    this through another few layers of bi-LSTMs/GRUs, and do a softmax over span
    start and span end.

    Parameters
    ----------
    vocab : ``Vocabulary``
    text_field_embedder : ``TextFieldEmbedder``
        Used to embed the ``question`` and ``context`` ``TextFields`` we get as input to the model.
    question_encoder : ``Seq2SeqEncoder``
        A encoder used for producing a (single-vector) representation of the question.
    context_encoder : ``Seq2SeqEncoder``
        The encoder for producing contextual representations of the context tokens, from the context
        word embeddings.
    matrix_attention : ``MatrixAttention``
        The attention to compute between the (tiled) question and the context.
    dropout : ``float``, optional (default=0.2)
        If greater than 0, we will apply dropout with this probability after all encoders (pytorch
        LSTMs do not apply dropout to their last layer).
    initializer : ``InitializerApplicator``, optional (default=``InitializerApplicator()``)
        Used to initialize the model parameters.
    regularizer : ``RegularizerApplicator``, optional (default=``None``)
        If provided, will be used to calculate the regularization penalty during training.
    """

    def __init__(
        self,
        vocab: Vocabulary,
        text_field_embedder: TextFieldEmbedder,
        question_encoder: Seq2SeqEncoder,
        context_encoder: Seq2SeqEncoder,
        matrix_attention: MatrixAttention,
        dropout: float = 0.2,
        initializer: InitializerApplicator = InitializerApplicator(),
        regularizer: Optional[RegularizerApplicator] = None,
    ) -> None:
        super().__init__(vocab, regularizer)

        self._text_field_embedder = text_field_embedder
        self._question_encoder = question_encoder
        self._context_encoder = context_encoder
        self._matrix_attention = matrix_attention
        if dropout > 0:
            self._dropout = torch.nn.Dropout(p=dropout)
        else:
            self._dropout = lambda x: x
        context_dim = context_encoder.get_output_dim()
        self._span_start_predictor = TimeDistributed(torch.nn.Linear(context_dim, 1))

        check_dimensions_match(
            self._text_field_embedder.get_output_dim(),
            self._question_encoder.get_input_dim(),
            "text embedding output dimension",
            "question encoder input dimension",
        )
        check_dimensions_match(
            self._text_field_embedder.get_output_dim(),
            self._context_encoder.get_input_dim(),
            "text embedding output dimension",
            "context encoder input dimension",
        )
        check_dimensions_match(
            self._context_encoder.get_output_dim(),
            self._question_encoder.get_output_dim(),
            "context encoder input dimension",
            "question encoder input dimension",
        )
        self._squad_metrics = SquadEmAndF1()
        initializer(self)

    def forward(  # type: ignore
        self,
        question: Dict[str, torch.LongTensor],
        context: Dict[str, torch.LongTensor],
        context_entity_mask: torch.FloatTensor,
        answer_as_passage_indices: torch.LongTensor = None,
        metadata: List[Dict[str, Any]] = None,
    ) -> Dict[str, torch.Tensor]:

        """
        Parameters
        ----------
        question : Dict[str, torch.LongTensor]
            From a ``TextField``. The question.
        context : Dict[str, torch.LongTensor]
            From a ``TextField``. The context.
        # TODO (nfliu): finish the rest of the docstring.
        metadata : ``List[Dict[str, Any]]``, optional
            If present, this should contain the question tokens, context tokens, original context
            text, answer text, and and the conversion mapping for entities (if applicable) for
            each instance in the batch.

        Returns
        -------
        An output dictionary consisting of:
        logits : torch.FloatTensor
            A tensor of shape ``(batch_size, num_labels)`` representing unnormalized log
            probabilities of the label.
        probabilities : torch.FloatTensor
            The result of ``softmax(logits)``.
        loss : torch.FloatTensor, optional
            A scalar loss to be optimised.
        """
        embedded_question = self._dropout(self._text_field_embedder(question))
        embedded_context = self._dropout(self._text_field_embedder(context))
        batch_size = embedded_question.size(0)
        question_mask = util.get_text_field_mask(question)
        context_mask = util.get_text_field_mask(context)

        # Shape: (batch_size, question_encoding_dim)
        encoded_question = self._question_encoder(embedded_question, question_mask)
        # Shape: (batch_size, passage_length, passage_encoding_dim)
        encoded_context = self._context_encoder(embedded_context, context_mask)
        # Shape: (batch_size, passage_length, question_length)
        context_question_similarity = self._matrix_attention(encoded_context, encoded_question)
        # Shape: (batch_size, passage_length, question_length)
        context_question_attention = util.masked_softmax(
            context_question_similarity, question_mask, memory_efficient=True
        )
        # Shape: (batch_size, passage_length, passage_encoding_dim)
        context_question_vectors = util.weighted_sum(encoded_question,
                                                     context_question_attention)
        # Shape: (batch_size, passage_length)
        answer_indices_logits = self._span_start_predictor(context_question_vectors).squeeze(-1)
        answer_indices_logits = util.replace_masked_values(answer_indices_logits,
                                                           context_entity_mask.bool(),
                                                           -1e32)
        answer_indices_log_probabilities = util.masked_log_softmax(answer_indices_logits, context_mask)

        output_dict = {
            "logits": answer_indices_logits,
            "probabilities": answer_indices_log_probabilities.exp()
        }

        # Compute the loss for training.
        if answer_as_passage_indices is not None:
            gold_passage_answer_indices = answer_as_passage_indices[:, :, 0]
            # Some indices are padded with index -1,
            # so we clamp those paddings to 0 and then mask after `torch.gather()`.
            gold_passage_answer_indices_mask = (gold_passage_answer_indices != -1)
            clamped_gold_passage_answer_indices = util.replace_masked_values(gold_passage_answer_indices,
                                                                             gold_passage_answer_indices_mask,
                                                                             0)
            # Shape: (batch_size, # of answer spans)
            log_likelihood_for_passage_answer_indices = \
                torch.gather(answer_indices_log_probabilities, 1, clamped_gold_passage_answer_indices)
            # For those padded indices, we set their log probabilities to be very small negative value
            log_likelihood_for_passage_answer_indices = \
                util.replace_masked_values(log_likelihood_for_passage_answer_indices,
                                           gold_passage_answer_indices_mask,
                                           -1e32)
            # Shape: (batch_size, )
            log_marginal_likelihood_for_passage_answer_indices = util.logsumexp(log_likelihood_for_passage_answer_indices)
            output_dict["loss"] = -log_marginal_likelihood_for_passage_answer_indices.mean()
            # Shape: (batch_size,)
            best_passage_answer_index = torch.max(answer_indices_logits, 1)[1]
            # Compute the metrics and add the tokenized input to the output.
            if metadata is not None:
                for i in range(batch_size):
                    answer_annotations = metadata[i].get('answer_text', "")
                    passage_tokens = metadata[i]["context_tokens"]
                    best_answer_str = passage_tokens[best_passage_answer_index[i].detach().cpu().numpy()]
                    if answer_annotations:
                        self._squad_metrics(best_answer_str, answer_annotations)
        return output_dict

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {
            "accuracy": self._squad_metrics.get_metric(reset)[0],
        }
