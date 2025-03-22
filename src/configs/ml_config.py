"""Parameters for ML models"""

from src.configs import names


###############################################################
#                                                             #
#                     EXPERIMENTS CONFIGS                     #
#                                                             #
###############################################################

EXPERIMENTS_CONFIGS = {
    0: {
        names.MODEL_TYPE: names.TRANSFORMER,
        # ARCHITECTURE
        names.EMBEDDING_DIMENSION: 64,
        names.CONTEXT_LENGTH: 24,
        names.NB_LAYERS: 4,
        names.NB_HEADS: 4,
        names.HEAD_OUTPUT_DIMENSION: 64,
        names.HEAD_SIZE: 16,  # = EMBEDDING_DIMENSION / NB_HEADS
        names.DROPOUT: 0.1,
        names.FEEDFORWARD_DIMENSION: 256,
        names.DEVICE: "cuda",
        # TRAINING
        names.NB_EPOCHS: 2,
        names.LEARNING_RATE: 1e-4,
        names.BATCH_SIZE: 32,
        names.NUM_WORKERS: 4,
        names.BETAS: (0.9, 0.98),
        names.EPSILON: 1e-9,
    },
    # Add more experiments as needed
}
