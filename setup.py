from setuptools import setup, find_packages

setup(
    name="llm_benchmarking",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "requests",
        "tqdm",
        "torch",
        "matplotlib",
        "seaborn",
        "datasets",
        "aiohttp",
    ],
    extras_require={
        "dev": [
            "pytest",
            "pytest-cov",
            "black",
            "isort",
            "flake8",
        ],
    },
    entry_points={
        "console_scripts": [
            "llm-benchmark=llm_benchmarking.cli.benchmark_cmd:main",
            "llm-analyze=llm_benchmarking.cli.analyze_cmd:main",
        ],
    },
    author="Shaun",
    author_email="shaun@example.com",
    description="A framework for evaluating Language Models via uncertainty quantification",
    keywords="llm, benchmarking, uncertainty, evaluation",
    url="https://github.com/yourusername/llm-benchmarking",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
    python_requires=">=3.8",
)
