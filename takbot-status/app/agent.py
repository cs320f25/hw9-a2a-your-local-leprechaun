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
from github import Github, GithubException
from datetime import datetime, timedelta

import os
import google.auth

_, project_id = google.auth.default()
os.environ["GOOGLE_CLOUD_PROJECT"] = project_id
os.environ["GOOGLE_CLOUD_LOCATION"] = "global"
os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "True"

# Agent Tools
def get_repo_info(repo_name: str) -> dict:
    """
    Get repository information
    
    Args:
        repo_name: Repository name in format 'owner/repo'
    
    Returns:
        Dictionary with repo information
    """
    github_token = os.environ.get("GITHUB_TOKEN")
    github_client = Github(github_token)
    
    try:
        repo = github_client.get_repo(repo_name)
        return {
            "name": repo.full_name,
            "description": repo.description,
            "stars": repo.stargazers_count,
            "forks": repo.forks_count,
            "open_issues": repo.open_issues_count,
            "default_branch": repo.default_branch,
            "last_updated": repo.updated_at.strftime("%Y-%m-%d %H:%M:%S"),
            "language": repo.language,
            "private": repo.private,
            "url": repo.html_url
        }
    except GithubException as e:
        return {"error": f"Failed to fetch repo info: {str(e)}"}

def get_recent_commits(repo_name: str, days: int = 7) -> list:
    """
    Get recent commits
    
    Args:
        repo_name: Repository name in format 'owner/repo'
        days: Number of days to look back (default: 7)
    
    Returns:
        List of recent commits
    """
    github_token = os.environ.get("GITHUB_TOKEN")
    github_client = Github(github_token)
    
    try:
        repo = github_client.get_repo(repo_name)
        since_date = datetime.now() - timedelta(days=days)
        commits = repo.get_commits(since=since_date)
        
        commit_list = []
        for commit in list(commits)[:10]:
            commit_list.append({
                "sha": commit.sha[:7],
                "author": commit.commit.author.name,
                "message": commit.commit.message.split('\n')[0],
                "date": commit.commit.author.date.strftime("%Y-%m-%d %H:%M:%S")
            })
        return commit_list
    except GithubException as e:
        return [{"error": f"Failed to fetch commits: {str(e)}"}]

def get_open_issues(repo_name: str) -> list:
    """
    Get open issues
    
    Args:
        repo_name: Repository name in format 'owner/repo'
    
    Returns:
        List of open issues
    """
    github_token = os.environ.get("GITHUB_TOKEN")
    github_client = Github(github_token)
    
    try:
        repo = github_client.get_repo(repo_name)
        issues = repo.get_issues(state='open')
        
        issue_list = []
        for issue in list(issues)[:20]:
            if not issue.pull_request:
                issue_list.append({
                    "number": issue.number,
                    "title": issue.title,
                    "state": issue.state,
                    "created_at": issue.created_at.strftime("%Y-%m-%d"),
                    "labels": [label.name for label in issue.labels],
                    "comments": issue.comments,
                    "url": issue.html_url
                })
        return issue_list
    except GithubException as e:
        return [{"error": f"Failed to fetch issues: {str(e)}"}]

def get_pull_requests(repo_name: str) -> list:
    """
    Get open pull requests
    
    Args:
        repo_name: Repository name in format 'owner/repo'
    
    Returns:
        List of open pull requests
    """
    github_token = os.environ.get("GITHUB_TOKEN")
    github_client = Github(github_token)
    
    try:
        repo = github_client.get_repo(repo_name)
        prs = repo.get_pulls(state='open')
        
        pr_list = []
        for pr in list(prs)[:10]:
            pr_list.append({
                "number": pr.number,
                "title": pr.title,
                "author": pr.user.login,
                "created_at": pr.created_at.strftime("%Y-%m-%d"),
                "mergeable": pr.mergeable,
                "comments": pr.comments,
                "review_comments": pr.review_comments,
                "url": pr.html_url
            })
        return pr_list
    except GithubException as e:
        return [{"error": f"Failed to fetch PRs: {str(e)}"}]

def get_project_structure(repo_name: str, path: str = "") -> list:
    """
    Get repository structure
    
    Args:
        repo_name: Repository name in format 'owner/repo'
        path: Path within repository (default: root)
    
    Returns:
        List of files and folders
    """
    github_token = os.environ.get("GITHUB_TOKEN")
    github_client = Github(github_token)
    
    try:
        repo = github_client.get_repo(repo_name)
        contents = repo.get_contents(path)
        
        structure = []
        for content in contents:
            structure.append({
                "name": content.name,
                "type": content.type,
                "path": content.path,
                "size": content.size if content.type == "file" else None
            })
        return structure
    except GithubException as e:
        return [{"error": f"Failed to fetch structure: {str(e)}"}]

def read_file(repo_name: str, file_path: str) -> dict:
    """
    Read file contents
    
    Args:
        repo_name: Repository name in format 'owner/repo'
        file_path: Path to the file in the repository
    
    Returns:
        File contents and metadata
    """
    github_token = os.environ.get("GITHUB_TOKEN")
    github_client = Github(github_token)
    
    try:
        repo = github_client.get_repo(repo_name)
        file_content = repo.get_contents(file_path)
        content = file_content.decoded_content.decode('utf-8')
        
        # Limit content size for LLM context
        if len(content) > 5000:
            content = content[:5000] + "\n... (truncated)"
        
        return {
            "path": file_path,
            "content": content,
            "size": file_content.size,
            "sha": file_content.sha
        }
    except GithubException as e:
        return {"error": f"Failed to read file: {str(e)}"}

root_agent = Agent(
    name="takbot_status",
    model="gemini-3-pro-preview",
    instruction="""
You are TakBot-Status, a GitHub status bot for the CS320 HW9 assignment.

Your job:
- You ONLY respond when you are directly asked for help in an issue.
- Issues live in the public repo: https://github.com/cs320f25/hw9-a2a-your-local-leprechaun

Your primary responsibilities are:
1. Monitor and report on the current state of repositories
2. Identify any problems, issues, or blockers that need attention
3. Summarize recent development activity and progress
4. Provide actionable insights and recommendations

Your specific rules:

1. Identify if the issue is talking to you:
   - The text references your name (StatusSprite) or clearly asks "the bot" for a status update.
   - If the issue does NOT appear to be directed at you, you must respond with NOTHING (the orchestrator will ignore you).

2. When it IS directed at you:
   - Use the GitHub tool to:
     - Inspect Luke's HW9 repo code (files/dirs, recent commits).
     - Inspect issues/PRs in Luke's HW9 repo to see recent work or blockers.
   - Summarize:
     - what parts of the implementation exist (which files, which features),
     - what looks incomplete or TODO,
     - any failing tests or obvious problems if visible,
     - and any open issues that look like blockers.

3. Be explicit and concrete:
   - Mention specific files, directories, and modules where work has been done.
   - Mention recent commit messages and what they imply about current progress.
   - If something is unclear or missing, say what’s missing and what you’d expect to see.

4. Style:
   - Write 2–4 short paragraphs.
   - Be honest about what you can and cannot see from the repo.
   - Do not try to guess beyond what the repo and its issues show.
   
When analyzing a repository:
- Start by getting basic repo information to understand the project
- Check recent commits to see development velocity
- Review open issues to identify problems
- Look at pull requests to see pending work
- If needed, examine specific files or project structure

Provide clear, well-organized reports that help users understand:
- Where the project currently stands
- What needs attention
- Recent progress and activity
- Any concerns or recommendations

If the request is not clearly for takbot-status, you must stay silent and not answer.
""",
    tools=[],
)

app = App(root_agent=root_agent, name="app")
