import pytest
from flaky import flaky
import numpy

from allennlp.common import Params
from allennlp.common.checks import ConfigurationError
from allennlp.common.testing import ModelTestCase
from allennlp.data.dataset import Batch
from allennlp.models import Model

from tests import FIXTURES_ROOT


class StanfordAttentiveReaderTest(ModelTestCase):
    def setUp(self):
        super().setUp()
        self.set_up_model(
            FIXTURES_ROOT / "stanford_attentive_reader" / "experiment.json",
            FIXTURES_ROOT / "data" / "cnn_dm.jsonl"
        )

    @flaky
    def test_forward_pass_runs_correctly(self):
        batch = Batch(self.instances)
        batch.index_instances(self.vocab)
        training_tensors = batch.as_tensor_dict()
        output_dict = self.model(**training_tensors)
        probs = output_dict["probabilities"][0].data.numpy()
        numpy.testing.assert_almost_equal(numpy.sum(probs, -1), numpy.array([1]))

    @flaky
    def test_model_can_train_save_and_load(self):
        self.ensure_model_can_train_save_and_load(self.param_file)

    def test_batch_predictions_are_consistent(self):
        self.ensure_batch_predictions_are_consistent()

    def test_mismatching_dimensions_throws_configuration_error(self):
        params = Params.from_file(self.param_file)
        # Make the encoder wrong - it should be 2 to match
        # the embedding dimension from the text_field_embedder.
        params["model"]["question_encoder"]["input_size"] = 10
        with pytest.raises(ConfigurationError):
            Model.from_params(vocab=self.vocab, params=params.pop("model"))
