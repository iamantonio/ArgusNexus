"""
ArgusNexus V4 Core - Setup Configuration

A Truth Engine-powered trading system where every decision is explainable
and every outcome is logged.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
readme_path = Path(__file__).parent / "README.md"
long_description = ""
if readme_path.exists():
    long_description = readme_path.read_text(encoding="utf-8")

# Read requirements
requirements_path = Path(__file__).parent / "requirements.txt"
requirements = []
if requirements_path.exists():
    with open(requirements_path) as f:
        requirements = [
            line.strip()
            for line in f
            if line.strip() and not line.startswith("#")
        ]

setup(
    name="argusnexus-v4-core",
    version="4.1.0",
    author="Tony",
    description="Truth Engine-powered trading system with Glass Box observability",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-repo/ArgusNexus-V4-Core",
    packages=find_packages(where="."),
    package_dir={"": "."},
    python_requires=">=3.9",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-asyncio>=0.21.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "isort>=5.12.0",
            "mypy>=1.0.0",
        ],
        "live": [
            "coinbase-advanced-py>=1.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "argus-backtest=scripts.run_backtest:main",
            "argus-paper=scripts.live_paper_trader:main",
            "argus-portfolio=scripts.live_portfolio_trader:main",
            "argus-api=src.api.main:run_server",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Financial and Insurance Industry",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Office/Business :: Financial :: Investment",
    ],
    keywords="trading, cryptocurrency, bitcoin, algo-trading, backtest",
    project_urls={
        "Bug Tracker": "https://github.com/your-repo/ArgusNexus-V4-Core/issues",
        "Documentation": "https://github.com/your-repo/ArgusNexus-V4-Core#readme",
    },
    include_package_data=True,
    package_data={
        "": ["*.yaml", "*.json", "*.html", "*.css", "*.js"],
    },
)
