"""
Configuration constants for the Multi-LLM Debate System.
"""

# LLM API Settings
DEFAULT_TIMEOUT_SEC = 2000
POST_CALL_DELAY_SEC = 5
LOG_INTERVAL_SEC = 10

# Concurrent Processing
MAX_CONCURRENCY = 5

# Problem Selection (for testing)
PROBLEMS_SKIP = 20
PROBLEMS_TAKE = 3

# Output Paths
DEFAULT_OUTPUT_DIR = "data/output"
PROBLEMS_PATH = "data/datasets/problems.json"