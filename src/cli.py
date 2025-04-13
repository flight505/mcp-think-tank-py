#!/usr/bin/env python3
"""
Command-line interface for MCP Think Tank
"""
import os
import sys
import argparse
import logging
from typing import Optional

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("mcp-think-tank")

# Try to import the model downloading function
try:
    from .tools.tasks import download_local_model
    CAN_DOWNLOAD_MODEL = True
except ImportError:
    CAN_DOWNLOAD_MODEL = False

def setup_command(args):
    """
    Run the setup process, including offering to download the local model
    
    Args:
        args: Command line arguments
    """
    print("\n===== MCP Think Tank Setup =====\n")
    
    # Create necessary directories
    base_dir = os.path.expanduser("~/.mcp-think-tank")
    os.makedirs(base_dir, exist_ok=True)
    os.makedirs(os.path.join(base_dir, "models"), exist_ok=True)
    
    print(f"Created data directory at: {base_dir}")
    
    # Setup local model if requested
    if args.with_local_model:
        if not CAN_DOWNLOAD_MODEL:
            print("\nError: Cannot download model - required packages not installed.")
            print("Please install the package with extra dependencies first:")
            print("pip install -e .[local-llm]")
            return
        
        print("\n----- Local Model Installation -----")
        print("\nThis will download the Gemma 3 1B model (~900MB) for local task parsing.")
        print("The model will be used as a fallback when Anthropic API is not available.")
        
        if not args.yes:
            answer = input("\nProceed with download? [y/N] ").lower()
            if answer not in ('y', 'yes'):
                print("Download cancelled.")
                return
        
        print("\nDownloading model...")
        success = download_local_model(force=args.force)
        
        if success:
            print("\nModel downloaded successfully!")
            print("The local model will be used as a fallback when Anthropic API is not available.")
        else:
            print("\nError downloading model.")
            print("You can try again later with:")
            print("  mcp-think-tank setup --with-local-model")
    else:
        print("\nSkipping local model installation.")
        print("You can install it later with:")
        print("  mcp-think-tank setup --with-local-model")
    
    # Setup Anthropic API key if requested
    if args.with_anthropic:
        print("\n----- Anthropic API Setup -----")
        
        if os.environ.get("ANTHROPIC_API_KEY"):
            print("\nAnthropic API key already set in environment.")
        else:
            print("\nTo use the Anthropic API for task parsing, you need to set the ANTHROPIC_API_KEY environment variable.")
            print("You can do this by adding the following to your shell profile (~/.bashrc, ~/.zshrc, etc.):")
            print('  export ANTHROPIC_API_KEY="your-api-key-here"')
    
    print("\n===== Setup Complete =====")
    print("\nYou can now run the MCP Think Tank server with:")
    print("  fastmcp install src/server.py")

def run_server_command(args):
    """
    Run the MCP Think Tank server
    
    Args:
        args: Command line arguments
    """
    print("Starting MCP Think Tank server...")
    
    # This will be implemented later when we have a complete server.py
    print("Server functionality not yet implemented.")
    print("Please use 'fastmcp install src/server.py' to install the server.")

def main():
    """Main entry point for the CLI"""
    parser = argparse.ArgumentParser(description="MCP Think Tank CLI")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Setup command
    setup_parser = subparsers.add_parser("setup", help="Set up MCP Think Tank")
    setup_parser.add_argument("--with-local-model", action="store_true", help="Download and install the local LLM model")
    setup_parser.add_argument("--with-anthropic", action="store_true", help="Configure Anthropic API integration")
    setup_parser.add_argument("--force", "-f", action="store_true", help="Force re-download even if model exists")
    setup_parser.add_argument("--yes", "-y", action="store_true", help="Answer yes to all prompts")
    
    # Server command (placeholder for now)
    server_parser = subparsers.add_parser("server", help="Run the MCP Think Tank server")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Run the appropriate command
    if args.command == "setup":
        setup_command(args)
    elif args.command == "server":
        run_server_command(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main() 