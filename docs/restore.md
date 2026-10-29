# 容器回收后的恢复

1. 克隆本私有仓库；校验 artifacts 下zip的SHA256，见 artifacts/SHA256SUMS。解压到新目录，原始路径作为历史会话标识保留。
2. 核对新预约、硬件、驱动、软件、模型与数据，记录新session。不要假定同一hostname就是同一运行环境。
3. 使用实验室提供的模型/数据；不从本仓库下载模型。插件固定SHA见版本文件。FlagGems官方目标commit/tree见版本文件；优先在新独立目录获取并验证该commit。
4. 如只能取得旧基点，可以在新副本应用 patches/official-tag.patch，补齐manifest的empty_files，并逐文件/完整tree核验。不得覆盖原安装。仅相同tag名称不足以验证版本。
5. 保留每次实际import路径、命令、环境与缓存位置。tools/compare_sources.py是有截止时间的诊断，需先按其前置条件配置；不能盲目运行来替代正式评测。
6. 独立审计已保存数据：解压后在解压根目录运行 tools/finalize_saved_result.py，传入raw、summary、accuracy-run-dir、environment及out参数。审计不调用模型；具体目录参见zip manifest。

本仓库不存SSH私钥、账号凭证或登录Cookie。恢复连接使用用户另行提供的认证；不复制原机器认证文件。
