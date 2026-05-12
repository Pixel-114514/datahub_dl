# Git 协作规范

本文档定义了项目的 Git 工作流程和提交规范，请严格遵守。

---

## 分支管理策略

### 主要分支

| 分支 | 用途 | 说明 |
|------|------|------|
| `main` | 生产环境分支 | 保持稳定，只接受经过测试的代码 |
| `develop` | 开发主分支 | 用于集成各个功能分支 |
| `release` | 发布分支 | 用于版本发布前的测试和修复 |

### 功能分支

| 分支类型 | 命名格式 | 示例 | 来源 | 合并目标 |
|----------|----------|------|------|----------|
| 功能开发 | `feature/功能名称` | `feature/add-navigation` | `develop` | `develop` |
| Bug 修复 | `bugfix/问题描述` | `bugfix/fix-motion-tracking` | `develop` | `develop` |
| 紧急修复 | `hotfix/紧急修复` | `hotfix/critical-crash-fix` | `main` | `main` + `develop` |

---

## 提交规范

### Commit Message 格式

```
<type>(<scope>): <subject>

<body>

<footer>
```

### Type 类型

| Type | 用途 |
|------|------|
| `feat` | 新功能 |
| `fix` | Bug 修复 |
| `docs` | 文档更新 |
| `style` | 代码格式调整（不影响功能） |
| `refactor` | 代码重构 |
| `perf` | 性能优化 |
| `test` | 测试相关 |
| `chore` | 构建过程或辅助工具的变动 |
| `ci` | CI/CD 配置修改 |

### 示例

```
feat(navigation): add obstacle avoidance algorithm

Implemented A* path planning with dynamic obstacle detection.
- Added collision detection module
- Integrated with sensor fusion system

Closes #123
```

---

## 工作流程

### 1. 创建功能分支

```bash
# 更新本地 develop 分支
git checkout develop
git pull origin develop

# 创建新功能分支
git checkout -b feature/your-feature-name
```

### 2. 开发与提交

```bash
# 查看修改
git status

# 添加修改文件
git add <file>

# 提交修改
git commit -m "feat(module): description"

# 定期推送到远程
git push origin feature/your-feature-name
```

### 3. 保持分支更新

```bash
# 定期同步 develop 分支的更新
git checkout develop
git pull origin develop
git checkout feature/your-feature-name
git rebase develop
```

### 4. 代码审查与合并

```bash
# 推送最新代码
git push origin feature/your-feature-name

# 在 GitLab/GitHub 上创建 Merge Request / Pull Request
# 等待代码审查通过后合并
```

---

## 代码审查规范

### 审查要点

- 代码功能是否符合需求
- 代码质量和可读性
- 是否有潜在的 Bug
- 是否符合项目编码规范
- 测试覆盖是否充分
- 文档是否完善

### 审查流程

1. 开发者创建 Merge Request (MR) / Pull Request (PR)
2. 指定至少一位审查者
3. 审查者进行代码审查，提出修改意见
4. 开发者根据意见修改代码
5. 审查通过后，由维护者合并代码

---

## 最佳实践

### 提交频率

- 小步提交，频繁推送
- 每个提交应该是一个完整的逻辑单元
- 避免一次性提交大量修改

### 代码冲突处理

```bash
# 拉取最新代码
git fetch origin

# 变基到最新的 develop
git rebase origin/develop

# 如果有冲突，解决冲突后
git add <resolved-files>
git rebase --continue

# 强制推送（谨慎使用）
git push origin feature/your-feature-name --force-with-lease
```

### 保持提交历史清晰

```bash
# 合并多个提交（交互式变基）
git rebase -i HEAD~n  # n 为要合并的提交数量

# 修改最后一次提交
git commit --amend
```

---

## 注意事项

- 不要直接在 `main` 或 `develop` 分支上开发
- 提交前确保代码能够编译通过
- 推送前运行测试确保功能正常
- 敏感信息（密码、密钥等）不要提交到仓库
- 大文件使用 Git LFS 管理
- 定期清理已合并的本地分支
