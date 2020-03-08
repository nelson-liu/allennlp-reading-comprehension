import pytest

from allennlp.common import Params
from allennlp.common.util import ensure_list
import numpy
from numpy.testing import assert_allclose

from allennlp_rc.dataset_readers import CnnDmReader
from tests import FIXTURES_ROOT


class TestCnnDailymailReader:
    @pytest.mark.parametrize("lazy", (True, False))
    def test_read_from_file(self, lazy):
        reader = CnnDmReader(lazy=lazy)
        instances = ensure_list(reader.read(FIXTURES_ROOT / "data" / "cnn_dm.jsonl"))
        assert len(instances) == 10

        instance = instances[0]
        assert set(instance.fields.keys()) == {
            "question",
            "context",
            "label_mask",
            "label",
            "metadata",
        }

        assert [t.text for t in instance["question"][:3]] == ["@entity8", "'s", "@placeholder"]
        assert [t.text for t in instance["context"][:3]] == ["@entity0", "safaris", "have"]
        assert [t.text for t in instance["context"][-3:]] == ["990", "per", "night"]
        assert_allclose(instance["label_mask"].array, numpy.ones(51))
        assert instance["label"].label == 9
        assert set(instance["metadata"].metadata.keys()) == {
            "original_context",
            "question_tokens",
            "context_tokens",
            "new_entity_ids",
            "answer_text"
        }
        # The question has one entity that the passage doesn't have.
        assert len(instance["metadata"].metadata["new_entity_ids"]) == 52
        assert instance["metadata"].metadata["answer_text"] == "@entity9"

    @pytest.mark.parametrize("lazy", (True, False))
    def test_read_no_relabel_entities(self, lazy):
        reader = CnnDmReader(relabel_entities=False, lazy=lazy)
        instances = ensure_list(reader.read(FIXTURES_ROOT / "data" / "cnn_dm.jsonl"))
        assert len(instances) == 10

        instance = instances[0]
        assert set(instance.fields.keys()) == {
            "question",
            "context",
            "label_mask",
            "label",
            "metadata",
        }

        assert [t.text for t in instance["question"][:3]] == ["@entity48", "'s", "@placeholder"]
        assert [t.text for t in instance["context"][:3]] == ["@entity1", "safaris", "have"]
        assert [t.text for t in instance["context"][-3:]] == ["990", "per", "night"]
        assert len(instance["label_mask"].array) == 283
        assert sum(instance["label_mask"].array) == 51
        assert instance["label"].label == 49
        assert set(instance["metadata"].metadata.keys()) == {
            "original_context",
            "question_tokens",
            "context_tokens",
            "answer_text"
        }
        assert instance["metadata"].metadata["answer_text"] == "@entity49"

    def test_can_build_from_params(self):
        reader = CnnDmReader.from_params(Params({}))
        assert reader._tokenizer.__class__.__name__ == "WhitespaceTokenizer"
        assert reader._token_indexers["tokens"].__class__.__name__ == "SingleIdTokenIndexer"
