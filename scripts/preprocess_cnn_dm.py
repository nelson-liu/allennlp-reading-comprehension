import argparse
import json
import logging
import os

from tqdm import tqdm

logger = logging.getLogger(__name__)


def main(dataset_path, output_path):
    dataset_path = os.path.abspath(dataset_path)
    instances = []
    logger.info(f"Loading questions from {dataset_path}")
    for instance_filename in tqdm(os.listdir(dataset_path)):
        instance_path = os.path.join(dataset_path, instance_filename)
        with open(instance_path) as instance_file:
            _, context, question, answer, _ = instance_file.read().split("\n\n")
            instances.append({"context": context,
                              "question": question,
                              "answer": answer})
    with open(output_path, "w") as output_file:
        for instance in instances:
            output_file.write(f"{json.dumps(instance)}\n")


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s " "- %(name)s - %(message)s", level=logging.INFO
    )
    parser = argparse.ArgumentParser(
        description=("Given a path to a folder with CNN / DailyMail RC dataset-formatted "
                     "questions, preprocess them into a more-compact JSONlines format. "
                     "Each line represents an instance, and has keys for the"
                     "string-formatted context, question, and answer."),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        required=True,
        help=("Path to folder with CNN or DailyMail questions (e.g., /cnn/training/).")
    )
    parser.add_argument(
        "--output-path",
        type=str,
        required=True,
        help=("Write output JSONlines-formatted dataset to this path")
    )
    args = parser.parse_args()
    main(args.dataset_path, args.output_path)
