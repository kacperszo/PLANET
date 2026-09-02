# PLANET, modern tier.
#
# Unlike SS-GNN this model has no compiled extensions, so nothing here is pinned to a
# torch ABI and the version below is a fidelity choice rather than a hard constraint:
# 2.5.1 is what the working venv resolved to. Python 3.13 matches it.
FROM docker.io/library/python:3.13-slim

ENV PIP_NO_CACHE_DIR=1 PYTHONUNBUFFERED=1 MPLBACKEND=Agg

RUN pip install --no-cache-dir \
    torch==2.5.1 \
    numpy==2.4.4 \
    scipy==1.17.1 \
    pandas \
    scikit-learn \
    matplotlib \
    rdkit==2026.3.1 \
    h5py \
    tqdm

# fail at build time rather than forty minutes into a run
RUN python -c "import torch, numpy, scipy, pandas, sklearn, matplotlib, rdkit, h5py, tqdm; \
print('env ok', torch.__version__)"

WORKDIR /work
COPY . /work
RUN python -c "from planet.model import PLANET; from planet.data import ProLigDataset; print('planet imports ok')"
