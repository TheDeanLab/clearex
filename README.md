<h1 align="center"> ClearEx
<h2 align="center"> Scalable Analytics for Cleared and Expanded Tissue Imaging.
</h2>
</h1>

[![Tests](https://github.com/TheDeanLab/clearex/actions/workflows/tests.yaml/badge.svg)](https://github.com/TheDeanLab/navigate/actions/workflows/push_checks.yaml)
[![codecov](https://codecov.io/gh/TheDeanLab/clearex/graph/badge.svg?token=ONldpMpLse)](https://codecov.io/gh/TheDeanLab/clearex)

**ClearEx** is an open source Python package for scalable analytics of cleared and expanded tissue imaging data. It relies heavily on next-generation file formats and cloud-based chunk computing to accelerate image analysis workflows, enabling tissue-scale computer vision and machine learning.

## Installation

We recommend installing ClearEx in a dedicated Anaconda environment:

```bash
conda create -n clearex python=3.11
conda activate clearex
pip install clearex
```

## Launching the CLI

Once installed and the environment is active, start the ClearEx command line interface by running:

```bash
clearex
```

Follow the prompts to perform image registration.
