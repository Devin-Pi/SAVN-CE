FROM rdpi/savnce:render-v1
# Base image recorded in environment.freeze.json. Existing Habitat/RLR remains unchanged.
COPY wheelhouse /tmp/wheelhouse
RUN python -m pip install --no-index --no-deps --find-links=/tmp/wheelhouse rtree==1.4.1 && rm -rf /tmp/wheelhouse
ENV HABITAT_SIM_LOG=quiet MAGNUM_LOG=quiet OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
WORKDIR /workspace/SAVN-CE
