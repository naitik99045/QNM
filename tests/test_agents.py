"""
Tests for Pre-Installed AI Agents and Tool Ecosystem.
"""

import os
import tempfile
from pathlib import Path
from qnm.agents.tools import FileSystemTool, TerminalTool, ToolRegistry
from qnm.agents.code_agent import CodeAgent
from qnm.agents.research_agent import ResearchAgent
from qnm.agents.shell_agent import ShellAgent


def test_filesystem_tool_lifecycle():
    with tempfile.TemporaryDirectory() as tmp_dir:
        fs = FileSystemTool(workspace_root=Path(tmp_dir))
        test_file = os.path.join(tmp_dir, "test.txt")

        # Write
        write_res = fs.write_file(test_file, "hello world\nline 2\n")
        assert "Successfully wrote" in write_res

        # Read
        read_res = fs.read_file(test_file)
        assert "hello world" in read_res

        # Edit
        edit_res = fs.edit_file(test_file, "hello world", "hello QNM")
        assert "Successfully replaced" in edit_res

        read_res2 = fs.read_file(test_file)
        assert "hello QNM" in read_res2

        # List
        list_res = fs.list_dir(tmp_dir)
        assert "test.txt" in list_res


def test_terminal_tool_execution():
    term = TerminalTool(timeout=5)
    res = term.run_command("python -c \"print('QNM_OK')\"")
    assert "[Exit Code]: 0" in res
    assert "QNM_OK" in res


def test_tool_registry():
    registry = ToolRegistry()
    registry.register(
        name="test_tool",
        description="A test tool",
        parameters={"x": {"type": "int"}},
        func=lambda x: f"result:{x * 2}"
    )
    assert registry.execute("test_tool", x=21) == "result:42"
    assert "Error: Unknown tool" in registry.execute("non_existent_tool")


def test_specialist_agents_instantiation_and_run():
    code_agent = CodeAgent(max_iterations=1)
    assert code_agent.name == "QNM-CodeAgent"
    res_code = code_agent.run("Verify file C:\\test.py")
    assert "final_answer" in res_code
    assert "C:\\test.py" in res_code["final_answer"] or "C:\\test.py" in res_code["goal"]

    research_agent = ResearchAgent(max_iterations=1)
    assert research_agent.name == "QNM-ResearchAgent"
    summary = research_agent.digest_document("This is a long specification for testing the concept compressor.")
    assert "[NCM Gist" in summary

    shell_agent = ShellAgent(max_iterations=1)
    assert shell_agent.name == "QNM-ShellAgent"
    res_shell = shell_agent.run("Check system status")
    assert "final_answer" in res_shell
