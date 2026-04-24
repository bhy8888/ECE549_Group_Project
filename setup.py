from setuptools import setup, find_packages

setup(
    name="ece549_apple_disease",
    version="0.1.0",
    description="Apple Disease Classification via ViT and Traditional CV",
    packages=find_packages(exclude=["tests*", "notebooks*"]),
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.1.0",
        "torchvision>=0.16.0",
        "transformers>=4.38.0",
        "timm>=0.9.12",
        "Pillow>=10.0.0",
        "numpy>=1.24.0",
        "scikit-learn>=1.3.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "opencv-python>=4.8.0",
        "PyYAML>=6.0",
        "tqdm>=4.65.0",
        "tensorboard>=2.14.0",
    ],
)
