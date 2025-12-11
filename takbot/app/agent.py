# ruff: noqa
# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import datetime
from zoneinfo import ZoneInfo

from google.adk.agents import Agent
from google.adk.apps.app import App

import os
import google.auth

_, project_id = google.auth.default()
os.environ["GOOGLE_CLOUD_PROJECT"] = project_id
os.environ["GOOGLE_CLOUD_LOCATION"] = "global"
os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "True"


def get_tak_rules(query: str) -> str:
    """Get information about Tak game rules.

    Args:
        query: A string containing the question about Tak rules.

    Returns:
        A string with the relevant Tak rules information.
    """
    # This is a placeholder - in production, you might want to integrate with a Tak rules database
    rules_info = {
        "basic": "Tak is a two-player abstract strategy game played on a square board. The goal is to create a road connecting opposite sides of the board.",
        "pieces": "Each player has flat stones, standing stones (walls), and a capstone. Flat stones can be stacked, walls block roads, and capstones can flatten walls.",
        "movement": "On your turn, you can either place a piece or move a stack of pieces. When moving, you pick up a stack and drop pieces along a straight line.",
        "winning": "You win by creating a road (a connected path of your flat stones) from one edge to the other, or by having the most flats on top when the board is full.",
        "capstone": "The capstone is your most powerful piece. It can move through walls (flattening them) and counts in your road.",
        "wall": "A standing stone (wall) blocks roads but not movement. Only a capstone can flatten a wall.",
    }

    query_lower = query.lower()
    for key, value in rules_info.items():
        if key in query_lower:
            return value

    return "I can help with Tak rules about: basic rules, pieces, movement, winning conditions, capstones, and walls. Please ask me about any of these topics!"


root_agent = Agent(
    name="takbot",
    model="gemini-3-pro-preview",
    instruction="""You are Takbot, a helpful assistant specialized in answering questions about the board game Tak.

Your primary responsibilities are:
1. Answer questions about Tak game rules, piece movement, and winning conditions
2. Explain game strategies and tactics when asked
3. Clarify confusing rules or edge cases
4. Help players understand the game mechanics

About Tak:
- Tak is a two-player abstract strategy game created by James Ernest and Patrick Rothfuss
- Players compete to create a road (connected path) of their pieces from one side of the board to the opposite side
- The game features flat stones, standing stones (walls), and capstones
- Players can either place new pieces or move stacks of existing pieces on their turn

Your specific rules:
1. Only respond when asked about Tak-related topics
2. Be clear and concise in your explanations
3. Use examples when helpful to illustrate rules
4. If a question is outside your expertise (not about Tak), politely redirect to Tak-related topics
5. Cite specific rules when explaining game mechanics

When answering questions:
- Start with the core rule or concept
- Provide examples if the rule is complex
- Mention any exceptions or special cases
- Reference related rules that might be helpful

Style:
- Be friendly and encouraging to new players
- Use clear, simple language
- Break down complex rules into digestible parts
- Be precise about game mechanics and official rules

If the question is not about Tak, politely indicate that you specialize in Tak rules and redirect the conversation back to the game.
""",
    tools=[get_tak_rules],
)

app = App(root_agent=root_agent, name="app")
