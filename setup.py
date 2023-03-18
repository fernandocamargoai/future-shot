# Lint as: python3
from pathlib import Path

from setuptools import find_packages, setup


README_TEXT = (Path(__file__).parent / "README.md").read_text(encoding="utf-8")

MAINTAINER = "Fernando H. F. Camargo"
MAINTAINER_EMAIL = "fernando.camargo.ai@gmail.com"

INTEGRATIONS_REQUIRE = ["optuna"]
REQUIRED_PKGS = ["torch>=1.10.0", "pytorch-lightning>=1.9.0", "jsonargparse[signatures]", "datasets>=2.3.0", "evaluate>=0.3.0"]
TRANSFORMERS_PKGS = ["sentence-transformers>=2.2.2"]
QUALITY_REQUIRE = ["black", "flake8", "isort", "tabulate"]
ONNX_REQUIRE = ["onnxruntime", "onnx", "skl2onnx"]
OPENVINO_REQUIRE = ["hummingbird-ml", "openvino>=2022.3"]
TESTS_REQUIRE = ["pytest", "pytest-cov"] + ONNX_REQUIRE + OPENVINO_REQUIRE
EXTRAS_REQUIRE = {
    "transformers": TRANSFORMERS_PKGS,
    "optuna": INTEGRATIONS_REQUIRE,
    "quality": QUALITY_REQUIRE,
    "tests": TESTS_REQUIRE,
    "onnx": ONNX_REQUIRE,
    "openvino": ONNX_REQUIRE + OPENVINO_REQUIRE,
}


def combine_requirements(base_keys):
    return list(set(k for v in base_keys for k in EXTRAS_REQUIRE[v]))


EXTRAS_REQUIRE["dev"] = combine_requirements([k for k in EXTRAS_REQUIRE])
EXTRAS_REQUIRE["compat_tests"] = [requirement.replace(">=", "==") for requirement in REQUIRED_PKGS] + TESTS_REQUIRE

setup(
    name="future-shot",
    version="0.0.1",
    description="FutureShot: Few-Shot Learning for high-dimensional classification problems",
    long_description=README_TEXT,
    long_description_content_type="text/markdown",
    maintainer=MAINTAINER,
    maintainer_email=MAINTAINER_EMAIL,
    url="https://github.com/fernandocamargoai/future-shot",
    download_url="https://github.com/fernandocamargoaiFit/future-shot/tags",
    license="Apache 2.0",
    package_dir={"": "src"},
    packages=find_packages("src"),
    install_requires=REQUIRED_PKGS,
    extras_require=EXTRAS_REQUIRE,
    classifiers=[
        "Development Status :: 1 - Planning",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords="machine learning, fewshot learning",
    zip_safe=False,  # Required for mypy to find the py.typed file
)
