from setuptools import find_namespace_packages, setup

setup(
    name="neural-chameleons",
    version="1.0.0",
    description="Neural Chameleons: Language Models Can Learn to Hide Their Thoughts from Activation Monitors",
    packages=find_namespace_packages(),
    author="Max McGuinness, Alex Serrano, Luke Bailey, Scott Emmons",
    author_email="max@max.rip, mail@alexserrano.org",
    url="https://github.com/mgm52/neural-chameleons",
    python_requires=">=3.10",
    install_requires=[
        # Base
        "transformers",
        "datasets",
        "accelerate",
        "scikit-learn",
        "hydra-core",
        "sae-lens",
        "numpy",
        "scipy",
        "einops",
        "pyyaml",
        "pydantic",
        # Visualization
        "matplotlib",
        "pandas",
        "seaborn",
        # Logging
        "wandb",
        "python-dotenv",
        "psutil",
        # Utils
        "tqdm",
        "jaxtyping",
        "sentencepiece",
        "bitsandbytes",
        "sentence-transformers",
        # Git dependencies
        "repe @ git+https://github.com/andyzoujm/representation-engineering.git@main",
        "eai-sparsify @ git+https://github.com/EleutherAI/sparsify@main",
        "strong_reject @ git+https://github.com/dsbowen/strong_reject.git@main",
    ],
    extras_require={
        "dev": ["pytest", "pre-commit", "ipykernel", "ipywidgets"],
    },
)
