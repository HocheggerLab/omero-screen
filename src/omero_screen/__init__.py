__version__ = "0.1.0"


import os
from pathlib import Path

from dotenv import load_dotenv


def set_env_vars() -> None:
    """
    Load environment variables based on the ENV variable.
    """
    # Determine the project root (adjust as necessary)
    project_root = Path(__file__).parent.parent.parent.resolve()

    # Path to the minimal .env file (optional)
    minimal_env_path = project_root / ".env"
    print(minimal_env_path)
    # Load the minimal .env file to get the ENV variable (if exists)
    if minimal_env_path.exists():
        load_dotenv(minimal_env_path)

    # Retrieve the ENV variable, default to 'development' if not set
    ENV = os.getenv("ENV", "development").lower()

    # Path to the environment-specific .env file
    env_specific_path = project_root / f".env.{ENV}"

    # Load the environment-specific .env file if it exists
    if env_specific_path.exists():
        load_dotenv(env_specific_path, override=True)
    else:
        print(
            f"Warning: {env_specific_path} not found. Using default configurations."
        )


set_env_vars()
