"""
This is a boilerplate pipeline 'finetuning_evaluation'
generated using Kedro 0.19.9
"""

from kedro.pipeline import Pipeline, pipeline, node
from .nodes import evaluate_finetuned_model


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            evaluate_finetuned_model,
            inputs=["params:finetune_eval_params"],
            outputs=None,
            name="evaluate_finetuned_model_node"
        )
    ])
