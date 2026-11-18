"""A server running Sellbird for others plugs its own model calls in through
``Brain(transport=...)``, and a home with no prompts folder still gets the
shipped prompts and skills."""

import asyncio
from pathlib import Path

import pytest

from sellbird.brain import Brain
from sellbird.workspace_home import Home, use_home


def test_a_home_without_prompts_gets_the_shipped_ones(tmp_path):
    nowhere = tmp_path / "nonexistent"
    home = Home.for_root(nowhere)
    with use_home(home):
        brain = Brain(state=None)
        assert "{{persona_company}}" in brain.load_prompt("writer")
        assert brain.load_skill("email_frameworks")
        assert brain.load_skill("product_knowledge_that_does_not_exist") == ""


def test_the_homes_own_prompt_wins(tmp_path):
    (tmp_path / "prompts").mkdir()
    (tmp_path / "prompts" / "writer.md").write_text("mine {{x}}")
    with use_home(Home.for_root(Path(tmp_path))):
        assert Brain(state=None).load_prompt("writer", x="1") == "mine 1"
