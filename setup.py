"""
Setup configuration for Chucklenet Predictor

This file defines the package installation and distribution for Chucklenet Predictor.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding='utf-8')

# Read requirements
requirements = []
requirements_path = this_directory / "requirements.txt"
if requirements_path.exists():
    with open(requirements_path, 'r', encoding='utf-8') as f:
        requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]

# Setup configuration
setup(
    name="chucklenet-predictor",
    version="1.0.0",
    author="Subho Das",
    author_email="subho.das@example.com",  # Replace with actual email
    description="Reverse Funny Strength Prediction Model with Multi-Modal Analysis",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Das-rebel/Chucklenet_predictor",
    project_urls={
        "Bug Tracker": "https://github.com/Das-rebel/Chucklenet_predictor/issues",
        "Documentation": "https://github.com/Das-rebel/Chucklenet_predictor/wiki",
        "Source Code": "https://github.com/Das-rebel/Chucklenet_predictor",
        "Research Paper": "https://arxiv.org/abs/XXXX.XXXX",  # Add paper when published
    },
    
    # Package configuration
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    
    # Include additional files
    include_package_data=True,
    package_data={
        "chucklenet_predictor": [
            "config/*.yaml",
            "models/*.json",
            "utils/*.py",
            "data/*.txt",
            "examples/*.py"
        ],
    },
    
    # Python version requirement
    python_requires=">=3.8",
    
    # Dependencies
    install_requires=requirements,
    
    # Optional dependencies
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "isort>=5.12.0",
            "pre-commit>=3.3.0",
        ],
        "audio": [
            "librosa>=0.10.0",
            "soundfile>=0.12.0",
            "audiocraft>=1.0.0",
        ],
        "evaluation": [
            "evaluate>=0.4.0",
            "sacrebleu>=2.3.0",
            "matplotlib>=3.7.0",
            "seaborn>=0.12.0",
        ],
        "training": [
            "accelerate>=0.21.0",
            "bitsandbytes>=0.41.0",
            "peft>=0.6.0",
            "deepspeed>=0.12.0",
        ],
        "visualization": [
            "matplotlib>=3.7.0",
            "seaborn>=0.12.0",
            "plotly>=5.15.0",
            "tensorboard>=2.13.0",
        ],
        "deployment": [
            "fastapi>=0.100.0",
            "uvicorn>=0.23.0",
            "streamlit>=1.25.0",
        ],
        "cloud": [
            "boto3>=1.28.0",
            "google-cloud-aiplatform>=1.25.0",
        ],
        "all": [
            # All optional dependencies combined
            "librosa>=0.10.0",
            "soundfile>=0.12.0",
            "audiocraft>=1.0.0",
            "evaluate>=0.4.0",
            "sacrebleu>=2.3.0",
            "matplotlib>=3.7.0",
            "seaborn>=0.12.0",
            "accelerate>=0.21.0",
            "bitsandbytes>=0.41.0",
            "peft>=0.6.0",
            "deepspeed>=0.12.0",
            "fastapi>=0.100.0",
            "uvicorn>=0.23.0",
            "streamlit>=1.25.0",
            "boto3>=1.28.0",
            "google-cloud-aiplatform>=1.25.0",
        ]
    },
    
    # Entry points for command-line tools
    entry_points={
        "console_scripts": [
            "chucklenet-predictor=chucklenet_predictor.cli:main",
            "chucklenet-train=chucklenet_predictor.training:main",
            "chucklenet-eval=chucklenet_predictor.evaluation:main",
            "chucklenet-demo=chucklenet_predictor.demo:main",
        ],
        "chucklenet.plugins": [
            "text_analysis=chucklenet_predictor.plugins.text_analyzer",
            "audio_analysis=chucklenet_predictor.plugins.audio_analyzer",
            "reverse_modeling=chucklenet_predictor.plugins.reverse_modeler",
        ],
    },
    
    # Classification
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Education",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "Topic :: Text Processing :: Linguistic",
        "Topic :: Multimedia :: Sound/Audio :: Analysis",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Games :: Entertainment",
    ],
    
    # Keywords
    keywords=[
        "humor", "funny", "laughter", "comedy", "entertainment", "nlp", "nlu",
        "machine-learning", "deep-learning", "transformers", "bert", "roberta",
        "audio-processing", "speech-recognition", "whisper", "multi-modal",
        "reverse-modeling", "content-generation", "ai", "artificial-intelligence",
        "quantitative-analysis", "linguistics", "psychology", "research",
        "entertainment", "content-creation", "script-writing", "stand-up-comedy"
    ],
    
    # Testing
    test_suite="tests",
    tests_require=[
        "pytest>=7.4.0",
        "pytest-cov>=4.1.0",
        "pytest-xdist>=3.3.0",
        "pytest-mock>=3.11.0",
    ],
    
    # Compilation
    zip_safe=False,
    
    # License and metadata
    license="MIT",
    license_files=["LICENSE"],
    
    # Installation
    platforms=["any"],
    
    
    
    # Data files
    data_files=[
        ("chucklenet_predictor/data", ["data/humor_dataset.csv", "models/config.yaml"]),
    ],
)