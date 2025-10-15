# Complete Git Mastery Guide
*Based on MLOps Course - Git Lecture*

## Git Fundamentals
| `rm .git or rm -rf .git` |🧨 2️⃣ Delete the old Git history | `" git init
git add .
git commit -m "Initial commit - my clean version   optional : git remote add origin https://github.com/<your-username>/<new-repo-name>.git
git branch -M main
git push -u origin main
"
` |
 
### Basic Configuration


| Command | Description | Example |
|---------|-------------|---------|
| `git config --global user.name "Name"` | Set global username | `git config --global user.name "John Doe"` |
| `git config --global user.email "email"` | Set global email | `git config --global user.email "john@example.com"` |
| `git config --list` | View all configurations | `git config --list` |

### Repository Setup

| Command | Description | Example |
|---------|-------------|---------|
| `git init` | Initialize new repository | `git init my-project` |
| `git clone <url>` | Clone existing repository | `git clone https://github.com/user/repo.git` |
| `git status` | Check repository status | `git status` |

## File Operations & Staging

### Basic File Workflow

| Command | Description | Example |
|---------|-------------|---------|
| `git add <file>` | Stage specific file | `git add script.py` |
| `git add .` | Stage all changes | `git add .` |
| `git add -A` | Stage all changes including deletions | `git add -A` |
| `git reset <file>` | Unstage file | `git reset script.py` |
| `git rm <file>` | Remove file from tracking | `git rm old_file.py` |

### Commit Operations

| Command | Description | Example |
|---------|-------------|---------|
| `git commit -m "message"` | Commit staged changes | `git commit -m "Add ML model"` |
| `git commit -am "message"` | Add and commit tracked files | `git commit -am "Update model"` |
| `git commit --amend` | Modify last commit | `git commit --amend -m "New message"` |

## Branch Management

### Branch Operations

| Command | Description | Example |
|---------|-------------|---------|
| `git branch` | List all branches | `git branch` |
| `git branch <name>` | Create new branch | `git branch feature-model` |
| `git checkout <branch>` | Switch to branch | `git checkout feature-model` |
| `git checkout -b <branch>` | Create and switch to branch | `git checkout -b experiment-1` |
| `git branch -d <branch>` | Delete branch (safe) | `git branch -d old-branch` |
| `git branch -D <branch>` | Force delete branch | `git branch -D failed-exp` |

### Branching Strategies for ML Projects

| Branch Type | Purpose | Naming Convention |
|-------------|---------|-------------------|
| Main/Master | Production ready code | `main` |
| Feature | New features/experiments | `feature/random-forest` |
| Experiment | ML experiments | `experiment/hyperparam-tuning` |
| Hotfix | Quick fixes | `hotfix/data-leak` |
| Release | Version releases | `release/v1.2.0` |

## Viewing History & Changes

### History Inspection

| Command | Description | Example |
|---------|-------------|---------|
| `git log` | Show commit history | `git log` |
| `git log --oneline` | Compact history view | `git log --oneline` |
| `git log --graph` | History with branch graph | `git log --graph --oneline` |
| `git show <commit>` | Show specific commit details | `git show abc123` |

### Difference Tracking

| Command | Description | Example |
|---------|-------------|---------|
| `git diff` | Show unstaged changes | `git diff` |
| `git diff --staged` | Show staged changes | `git diff --staged` |
| `git diff <commit1> <commit2>` | Compare two commits | `git diff abc123 def456` |
| `git diff <branch1>..<branch2>` | Compare two branches | `git diff main..feature` |

## Merging & Rebasing

### Merge Strategies

| Command | Description | Example | Use Case |
|---------|-------------|---------|----------|
| `git merge <branch>` | Fast-forward merge | `git merge feature` | Simple feature completion |
| `git merge --no-ff <branch>` | Create merge commit | `git merge --no-ff feature` | Preserve branch history |
| `git merge --squash <branch>` | Squash all commits | `git merge --squash feature` | Clean history |

### Rebase Operations

| Command | Description | Example | When to Use |
|---------|-------------|---------|-------------|
| `git rebase <branch>` | Rebase current branch | `git rebase main` | Update feature branch |
| `git rebase -i <commit>` | Interactive rebase | `git rebase -i HEAD~3` | Clean up commit history |
| `git rebase --abort` | Abort rebase | `git rebase --abort` | Cancel ongoing rebase |

## Undoing Changes

### Reset Operations

| Command | Description | Example | Risk Level |
|---------|-------------|---------|------------|
| `git reset --soft <commit>` | Keep changes staged | `git reset --soft HEAD~1` | Low |
| `git reset --mixed <commit>` | Keep changes unstaged | `git reset --mixed HEAD~1` | Medium |
| `git reset --hard <commit>` | Discard all changes | `git reset --hard HEAD~1` | **HIGH** |

### Revert & Checkout

| Command | Description | Example |
|---------|-------------|---------|
| `git revert <commit>` | Create undo commit | `git revert abc123` |
| `git checkout -- <file>` | Discard file changes | `git checkout -- model.py` |
| `git checkout <commit>` | View old commit (detached HEAD) | `git checkout abc123` |

## Remote Operations

### Remote Repository Management

| Command | Description | Example |
|---------|-------------|---------|
| `git remote add <name> <url>` | Add remote repository | `git remote add origin https://github.com/user/repo.git` |
| `git remote -v` | View remote connections | `git remote -v` |
| `git push -u <remote> <branch>` | Push and set upstream | `git push -u origin main` |
| `git push` | Push to upstream | `git push` |
| `git pull` | Fetch and merge | `git pull` |
| `git fetch` | Download without merge | `git fetch` |

## Stashing Changes

### Stash Operations

| Command | Description | Example |
|---------|-------------|---------|
| `git stash` | Save changes temporarily | `git stash` |
| `git stash save "message"` | Stash with message | `git stash save "WIP: model training"` |
| `git stash list` | List all stashes | `git stash list` |
| `git stash apply` | Apply last stash | `git stash apply` |
| `git stash pop` | Apply and remove stash | `git stash pop` |
| `git stash drop` | Delete specific stash | `git stash drop stash@{0}` |

## Tagging for ML Versioning

### Version Management

| Command | Description | Example |
|---------|-------------|---------|
| `git tag <tagname>` | Create lightweight tag | `git tag v1.0` |
| `git tag -a <tagname> -m "message"` | Create annotated tag | `git tag -a v1.1 -m "Random Forest v1.1"` |
| `git tag` | List all tags | `git tag` |
| `git push --tags` | Push all tags to remote | `git push --tags` |

## ML-Specific Git Workflow

### Commit Message Convention for ML

| Prefix | Purpose | Example |
|--------|---------|---------|
| `feat:` | New feature | `feat: add neural network model` |
| `fix:` | Bug fix | `fix: resolve data leakage` |
| `docs:` | Documentation | `docs: update model API docs` |
| `experiment:` | ML experiment | `experiment: test different optimizers` |
| `data:` | Data changes | `data: add new training dataset` |
| `model:` | Model changes | `model: update hyperparameters` |

### ML Project Structure with Git
# Complete Git Mastery Guide
*Based on MLOps Course - Git Lecture*

## Git Fundamentals

### Basic Configuration

| Command | Description | Example |
|---------|-------------|---------|
| `git config --global user.name "Name"` | Set global username | `git config --global user.name "John Doe"` |
| `git config --global user.email "email"` | Set global email | `git config --global user.email "john@example.com"` |
| `git config --list` | View all configurations | `git config --list` |

### Repository Setup

| Command | Description | Example |
|---------|-------------|---------|
| `git init` | Initialize new repository | `git init my-project` |
| `git clone <url>` | Clone existing repository | `git clone https://github.com/user/repo.git` |
| `git status` | Check repository status | `git status` |

## File Operations & Staging

### Basic File Workflow

| Command | Description | Example |
|---------|-------------|---------|
| `git add <file>` | Stage specific file | `git add script.py` |
| `git add .` | Stage all changes | `git add .` |
| `git add -A` | Stage all changes including deletions | `git add -A` |
| `git reset <file>` | Unstage file | `git reset script.py` |
| `git rm <file>` | Remove file from tracking | `git rm old_file.py` |

### Commit Operations

| Command | Description | Example |
|---------|-------------|---------|
| `git commit -m "message"` | Commit staged changes | `git commit -m "Add ML model"` |
| `git commit -am "message"` | Add and commit tracked files | `git commit -am "Update model"` |
| `git commit --amend` | Modify last commit | `git commit --amend -m "New message"` |

## Branch Management

### Branch Operations

| Command | Description | Example |
|---------|-------------|---------|
| `git branch` | List all branches | `git branch` |
| `git branch <name>` | Create new branch | `git branch feature-model` |
| `git checkout <branch>` | Switch to branch | `git checkout feature-model` |
| `git checkout -b <branch>` | Create and switch to branch | `git checkout -b experiment-1` |
| `git branch -d <branch>` | Delete branch (safe) | `git branch -d old-branch` |
| `git branch -D <branch>` | Force delete branch | `git branch -D failed-exp` |

### Branching Strategies for ML Projects

| Branch Type | Purpose | Naming Convention |
|-------------|---------|-------------------|
| Main/Master | Production ready code | `main` |
| Feature | New features/experiments | `feature/random-forest` |
| Experiment | ML experiments | `experiment/hyperparam-tuning` |
| Hotfix | Quick fixes | `hotfix/data-leak` |
| Release | Version releases | `release/v1.2.0` |

## Viewing History & Changes

### History Inspection

| Command | Description | Example |
|---------|-------------|---------|
| `git log` | Show commit history | `git log` |
| `git log --oneline` | Compact history view | `git log --oneline` |
| `git log --graph` | History with branch graph | `git log --graph --oneline` |
| `git show <commit>` | Show specific commit details | `git show abc123` |

### Difference Tracking

| Command | Description | Example |
|---------|-------------|---------|
| `git diff` | Show unstaged changes | `git diff` |
| `git diff --staged` | Show staged changes | `git diff --staged` |
| `git diff <commit1> <commit2>` | Compare two commits | `git diff abc123 def456` |
| `git diff <branch1>..<branch2>` | Compare two branches | `git diff main..feature` |

## Merging & Rebasing

### Merge Strategies

| Command | Description | Example | Use Case |
|---------|-------------|---------|----------|
| `git merge <branch>` | Fast-forward merge | `git merge feature` | Simple feature completion |
| `git merge --no-ff <branch>` | Create merge commit | `git merge --no-ff feature` | Preserve branch history |
| `git merge --squash <branch>` | Squash all commits | `git merge --squash feature` | Clean history |

### Rebase Operations

| Command | Description | Example | When to Use |
|---------|-------------|---------|-------------|
| `git rebase <branch>` | Rebase current branch | `git rebase main` | Update feature branch |
| `git rebase -i <commit>` | Interactive rebase | `git rebase -i HEAD~3` | Clean up commit history |
| `git rebase --abort` | Abort rebase | `git rebase --abort` | Cancel ongoing rebase |

## Undoing Changes

### Reset Operations

| Command | Description | Example | Risk Level |
|---------|-------------|---------|------------|
| `git reset --soft <commit>` | Keep changes staged | `git reset --soft HEAD~1` | Low |
| `git reset --mixed <commit>` | Keep changes unstaged | `git reset --mixed HEAD~1` | Medium |
| `git reset --hard <commit>` | Discard all changes | `git reset --hard HEAD~1` | **HIGH** |

### Revert & Checkout

| Command | Description | Example |
|---------|-------------|---------|
| `git revert <commit>` | Create undo commit | `git revert abc123` |
| `git checkout -- <file>` | Discard file changes | `git checkout -- model.py` |
| `git checkout <commit>` | View old commit (detached HEAD) | `git checkout abc123` |

## Remote Operations

### Remote Repository Management

| Command | Description | Example |
|---------|-------------|---------|
| `git remote add <name> <url>` | Add remote repository | `git remote add origin https://github.com/user/repo.git` |
| `git remote -v` | View remote connections | `git remote -v` |
| `git push -u <remote> <branch>` | Push and set upstream | `git push -u origin main` |
| `git push` | Push to upstream | `git push` |
| `git pull` | Fetch and merge | `git pull` |
| `git fetch` | Download without merge | `git fetch` |

## Stashing Changes

### Stash Operations

| Command | Description | Example |
|---------|-------------|---------|
| `git stash` | Save changes temporarily | `git stash` |
| `git stash save "message"` | Stash with message | `git stash save "WIP: model training"` |
| `git stash list` | List all stashes | `git stash list` |
| `git stash apply` | Apply last stash | `git stash apply` |
| `git stash pop` | Apply and remove stash | `git stash pop` |
| `git stash drop` | Delete specific stash | `git stash drop stash@{0}` |

## Tagging for ML Versioning

### Version Management

| Command | Description | Example |
|---------|-------------|---------|
| `git tag <tagname>` | Create lightweight tag | `git tag v1.0` |
| `git tag -a <tagname> -m "message"` | Create annotated tag | `git tag -a v1.1 -m "Random Forest v1.1"` |
| `git tag` | List all tags | `git tag` |
| `git push --tags` | Push all tags to remote | `git push --tags` |

## ML-Specific Git Workflow

### Commit Message Convention for ML

| Prefix | Purpose | Example |
|--------|---------|---------|
| `feat:` | New feature | `feat: add neural network model` |
| `fix:` | Bug fix | `fix: resolve data leakage` |
| `docs:` | Documentation | `docs: update model API docs` |
| `experiment:` | ML experiment | `experiment: test different optimizers` |
| `data:` | Data changes | `data: add new training dataset` |
| `model:` | Model changes | `model: update hyperparameters` |

