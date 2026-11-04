# Transformer From Zero · 365 Days

> 2026-11 → 2027-10 ｜自己推公式、写模块、做测试、训练模型、记录实验。

这是一个**可运行的学习型 Python 仓库起点**，不是已经实现好的 Transformer。首周把环境、数学基础和测试规范跑通；后续逐步亲手补全注意力、Encoder–Decoder、GPT、训练和优化。

## 先确定规则

1. **核心模型模块自己写**：可以用 PyTorch Tensor / Autograd / Optimizer，但不用 `nn.Transformer` 或 `nn.MultiheadAttention` 直接代替个人实现。
2. **先公式再代码**：每个新模块先写输入、输出、维度和数学定义。
3. **先测试再训练**：核查形状、数值、梯度、因果遮挡和微型数据过拟合。
4. **可以参考，但不复制提交**：参考实现用来比较输出和排障，不作为自己代码的替身。
5. **每周留下可检查产物**：代码、测试、笔记和一次 Git 提交。

## 我们选 VS Code

推荐 **VS Code + Python 扩展 + Pylance + Git**。适合 Windows 本地编写和调试、远程 AutoDL 训练。PyCharm 也完全可以打开此文件夹；项目没有绑定 IDE。

### 1. Windows 11（PowerShell）

建议安装 Python **3.11** 和 Git。然后在 VS Code 中使用「文件 → 打开文件夹」打开解压后的 `transformer-from-zero`：

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python scripts/verify_setup.py
python -m pytest -q
```

如果 PowerShell 因执行策略阻止激活，不必修改系统策略，可以直接运行：

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe scripts/verify_setup.py
.\.venv\Scripts\python.exe -m pytest -q
```

然后在 VS Code 中选择 **Python: Select Interpreter** → `.venv` 里的 Python。

**4060 显卡提示**：`requirements.txt` 不会保证安装 CUDA 版本的 PyTorch。先用 CPU 确认全部测试通过；若 `verify_setup.py` 提示 `CUDA available: False`，参照 [PyTorch 官方安装页](https://pytorch.org/get-started/locally/) 选择 Windows / Pip / Python / 对应 CUDA 版本安装适合你的 PyTorch。不要盲目安装旧 CUDA wheel。第一周任务用 CPU 足够。

### 2. AutoDL / Linux（训练阶段可用）

不要改动正在用于科研项目的环境，建议单独建环境。以下命令**全部前台运行**：

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python scripts/verify_setup.py
python -m pytest -q
```

若实例没有 `python3.11`，使用独立 conda 环境或现有可用 Python 3.10+，安装前核对版本。GPU 版 PyTorch 依照该实例的驱动与官方安装指引选取。

### 3. Git 初始化（解压后执行一次）

本压缩包提供的是 **Git-ready 项目文件**，并没有代替你注册或创建远端 GitHub 仓库：

```powershell
git init -b main
git add .
git commit -m "chore: bootstrap transformer-from-zero"
```

如遇 `Author identity unknown`，先按照提示配置你自己的 Git 用户名和邮箱。之后可在 GitHub 创建空仓库并设置 `origin` 后推送；不必上传数据集、模型权重或 `.venv`。

## 第一天怎么开始

1. 安装并激活独立环境，执行 `python scripts/verify_setup.py`。
2. 执行 `python -m pytest -q`：基础 smoke tests 应通过。
3. 阅读 `docs/02_week01_tasks.md`。
4. 打开 `exercises/week01.py`，自己填完三个 `NotImplementedError`。
5. 执行 `python -m pytest -q -m exercise`：**刚解压时此命令会失败，是有意设计的**。三个练习完成后应全部通过。
6. 在 `docs/weekly-notes/` 写下张量维度、公式与遇到的错误。

## 项目目录

```text
transformer-from-zero/
├─ .vscode/                  # VS Code 调试、测试任务和扩展建议
├─ docs/
│  ├─ 00_project_charter.md
│  ├─ 01_year_roadmap.md
│  ├─ 02_week01_tasks.md
│  ├─ experiment_log_template.md
│  └─ weekly-notes/
├─ exercises/
│  └─ week01.py              # 第 1 周待亲手实现的数学基础
├─ src/tfz/
│  ├─ __init__.py
│  ├─ utils.py               # 初始工具函数
│  └─ models/                # 后续逐步加入 Attention 等模块
├─ scripts/
│  └─ verify_setup.py
├─ tests/
│  ├─ test_smoke.py
│  └─ test_week01_exercises.py
├─ data/                     # 不纳入 Git 的本地数据
├─ checkpoints/              # 不纳入 Git 的模型权重
├─ results/                  # 不纳入 Git 的训练结果
├─ .gitignore
├─ pyproject.toml
└─ requirements.txt
```

## 月度验收节奏

见 [12 个月路线图](docs/01_year_roadmap.md)。第 1 个月暂不写完整 Transformer；避免“仓库看上去很完整，自己却讲不清模型”的情况。

## 资源

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [The Annotated Transformer](https://nlp.seas.harvard.edu/annotated-transformer/)
- [Karpathy: Let's build GPT](https://www.youtube.com/watch?v=kCc8FmEb1nY)
- [nanoGPT（对照参考，不是本项目起点）](https://github.com/karpathy/nanoGPT)

## License

尚未选定开源协议。决定公开发布时，按自己的需求添加 LICENSE 文件。
