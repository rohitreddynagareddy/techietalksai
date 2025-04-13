import sys
from .server import serve

def main():
    """MCP Hello World"""
    import argparse
    import asyncio

    # Open the log file in write mode
    log_file = open("/app/logs/logs.txt", "w")
    # Redirect stdout and stderr to the log file
    sys.stdout = log_file
    sys.stderr = log_file

    parser = argparse.ArgumentParser(
        description="give a model the ability to run a function"
    )

    args = parser.parse_args()
    asyncio.run(serve())

    # Close the log file when done
    log_file.close()

if __name__ == "__main__":
    main()
