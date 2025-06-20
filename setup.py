from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="pytorch-hmm",
    version="0.2.0",
    author="Speech Synthesis Engineer",
    author_email="crlotwhite@users.noreply.github.com",
    description="Advanced PyTorch implementation of Hidden Markov Models for speech synthesis and sequence modeling",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/crlotwhite/pytorch_hmm",
    project_urls={
        "Bug Reports": "https://github.com/crlotwhite/pytorch_hmm/issues",
        "Source": "https://github.com/crlotwhite/pytorch_hmm",
        "Documentation": "https://github.com/crlotwhite/pytorch_hmm/wiki",
    },
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Education",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Multimedia :: Sound/Audio :: Speech",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Framework :: Jupyter",
    ],
    keywords=[
        "hmm", "hidden-markov-model", "speech-synthesis", "tts", "pytorch", 
        "neural-hmm", "semi-markov", "dtw", "ctc", "sequence-modeling",
        "speech-processing", "alignment", "duration-modeling"
    ],
    python_requires=">=3.7",
    install_requires=[
        "torch>=1.8.0",
        "numpy>=1.19.0",
        "tqdm>=4.60.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=3.0",
            "pytest-benchmark>=4.0",
            "black>=22.0",
            "isort>=5.0",
            "flake8>=4.0",
            "mypy>=0.910",
        ],
        "visualization": [
            "matplotlib>=3.3.0",
            "seaborn>=0.11.0",
            "plotly>=5.0.0",
        ],
        "audio": [
            "librosa>=0.8.0",
            "soundfile>=0.10.0",
            "scipy>=1.7.0",
        ],
        "benchmarks": [
            "psutil>=5.8.0",
            "memory-profiler>=0.60.0",
            "line-profiler>=3.0.0",
        ],
        "notebooks": [
            "jupyter>=1.0.0",
            "ipython>=7.0.0",
            "ipywidgets>=7.6.0",
        ],
        "all": [
            "pytest>=6.0", "pytest-cov>=3.0", "pytest-benchmark>=4.0",
            "black>=22.0", "isort>=5.0", "flake8>=4.0", "mypy>=0.910",
            "matplotlib>=3.3.0", "seaborn>=0.11.0", "plotly>=5.0.0",
            "librosa>=0.8.0", "soundfile>=0.10.0", "scipy>=1.7.0",
            "psutil>=5.8.0", "memory-profiler>=0.60.0", "line-profiler>=3.0.0",
            "jupyter>=1.0.0", "ipython>=7.0.0", "ipywidgets>=7.6.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "pytorch-hmm-demo=examples.advanced_features_demo:main",
            "pytorch-hmm-test=tests.test_integration:run_comprehensive_tests",
        ],
    },
    package_data={
        "pytorch_hmm": ["*.py"],
        "examples": ["*.py"],
        "tests": ["*.py"],
    },
    include_package_data=True,
    zip_safe=False,
    platforms=["any"],
)
