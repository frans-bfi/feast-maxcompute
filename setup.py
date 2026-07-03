from setuptools import setup, find_packages

setup(
    name="feast-maxcompute",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "feast==0.63.0",
        "pyodps>=0.11",
        "pandas>=1.4",
        "pyarrow>=11",
    ],
    entry_points={
        "feast.data_source": [
            "maxcompute = feast_maxcompute.maxcompute_source:MaxComputeSource",
        ],
        "feast.offline_store": [
            "maxcompute = feast_maxcompute.maxcompute_offline_store:MaxComputeOfflineStore",
        ],
    },
)