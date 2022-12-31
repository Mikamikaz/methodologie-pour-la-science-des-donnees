"""
This is a boilerplate pipeline 'data_science'
generated using Kedro 0.18.4
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes_packages.code1 import premier_modele
from .nodes_packages.code2 import deuxieme_modele
from .nodes_packages.code3 import troisieme_modele


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=premier_modele,
                inputs=[],
                outputs="history_df1",
                name="premier_modele",
            ),
            node(
                func=deuxieme_modele,
                inputs=[],
                outputs='history_df2',
                name="deuxieme_modele",
            ),
            node(
                func=troisieme_modele,
                inputs=[],
                outputs='history_df3',
                name="troisieme_modele",
            )
        ]
    )
