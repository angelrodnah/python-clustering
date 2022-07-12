from setuptools import setup
from pathlib import Path

if __name__ == "__main__":
    current_directory = Path(__file__).parent
    setup(
        long_description = (current_directory / "README.md").read_text(),
        long_description_content_type="text/markdown"
    )
