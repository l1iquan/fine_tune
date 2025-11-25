# GitHub 推送指南

## 步骤1：在GitHub上创建新仓库

1. 登录到你的GitHub账户
2. 点击右上角的 "+" 图标，选择 "New repository"
3. 填写仓库信息：
   - **Repository name**: `qwen3-14b-testcase-generator` (或你喜欢的名称)
   - **Description**: `基于Qwen3-14B的测试用例生成助手项目`
   - **Public** 或 **Private** (根据你的需求选择)
   - **不要**勾选 "Initialize this repository with a README" (因为我们已经有了)
4. 点击 "Create repository"

## 步骤2：连接本地仓库到GitHub

在命令行中执行以下命令（将 `YOUR_USERNAME` 替换为你的GitHub用户名）：

```bash
# 添加远程仓库
git remote add origin https://github.com/YOUR_USERNAME/qwen3-14b-testcase-generator.git

# 推送代码到GitHub
git push -u origin master
```

或者如果你使用main分支：

```bash
git push -u origin main
```

## 步骤3：验证推送成功

1. 刷新你的GitHub仓库页面
2. 你应该能看到所有文件已经上传
3. README.md文件会自动显示在仓库主页

## 步骤4：后续更新（可选）

当你对项目进行修改后，可以使用以下命令更新GitHub：

```bash
# 添加所有更改
git add .

# 提交更改
git commit -m "描述你的更改"

# 推送到GitHub
git push
```

## 注意事项

- 首次推送可能需要输入GitHub用户名和密码（或使用个人访问令牌）
- 如果遇到认证问题，可以设置SSH密钥或使用GitHub CLI
- 确保你的Git配置了正确的用户名和邮箱：
  ```bash
  git config --global user.name "你的GitHub用户名"
  git config --global user.email "你的邮箱@example.com"
  ```

## 仓库链接示例

成功推送后，你的仓库地址将是：
`https://github.com/YOUR_USERNAME/qwen3-14b-testcase-generator`

现在你的项目已经准备好与其他人分享了！
