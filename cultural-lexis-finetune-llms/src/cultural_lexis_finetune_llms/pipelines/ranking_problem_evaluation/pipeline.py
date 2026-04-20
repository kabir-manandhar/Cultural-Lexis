"""
This is a boilerplate pipeline 'ranking_problem_evaluation'
generated using Kedro 0.19.9
"""

from kedro.pipeline import Pipeline, pipeline, node

from .nodes import evaluate_ranking_problem


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            evaluate_ranking_problem,
            inputs=["params:ranking_problem_eval_params"],
            outputs="ranking_problem_eval_results",
            name="evaluate_ranking_problem_node"
        )
    ])
