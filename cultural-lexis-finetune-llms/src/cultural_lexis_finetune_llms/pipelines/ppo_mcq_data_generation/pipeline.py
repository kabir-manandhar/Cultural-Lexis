"""
This is a boilerplate pipeline 'ppo_mcq_data_generation'
generated using Kedro 0.19.9
"""

from kedro.pipeline import Pipeline, pipeline, node

from .nodes import generate_mcq_data, mcq_test_model_output, generate_ranking_data, ranking_test_model_output


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        # node( # deprecated 
        #     func=generate_mcq_data,
        #     inputs=[
        #         "params:ppo_mcq_params"
        #     ],
        #     outputs=None,
        #     name="generate_mcq_data_node"
        # ),
        node(
            func=generate_ranking_data,
            inputs=[
                "params:ppo_ranking_params"
            ],
            outputs=None,
            name="generate_ranking_data_node"
        ),
        
        
        # node( # deprecated
        #     func=mcq_test_model_output,
        #     inputs=[
        #         "params:ppo_mcq_params",
        #     ],
        #     outputs=None,
        #     name="mcq_test_model_output_node"
        # ),
        
        
        
        
        # node(
        #     func=ranking_test_model_output,
        #     inputs=[
        #         "params:ppo_ranking_params",
        #         "params:ppo_mcq_params.model_params",
        #     ],
        #     outputs=None,
        #     name="ranking_test_model_output_node"
        # )
    ])
