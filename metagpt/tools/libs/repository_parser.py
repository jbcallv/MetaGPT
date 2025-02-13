import subprocess
from metagpt.tools.tool_registry import register_tool

@register_tool(tags=["repository", "parsing", "parser", "github"], include_functions=["__init__", "clone_repository"])
class RepositoryParser:
    """
    A repository parsing tool that clones a GitHub repository and parses relevant code components into memory
    """
    def __init__(self):
        print("In the init of repo parser")

    @staticmethod
    def clone_repository(repo_url, target_dir: str = "."):
        subprocess.run(['git', 'clone', repo_url, target_dir], check=True)