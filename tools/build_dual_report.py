"""Render a review PDF from audited results; never invents a MetaX pass."""
import argparse
import html
import json
from pathlib import Path

from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table, TableStyle, Image
from reportlab.graphics.shapes import Drawing, Rect, String, Line

NAVY = colors.HexColor('#173d55')
TEAL = colors.HexColor('#168078')
GREY = colors.HexColor('#617487')
PALE = colors.HexColor('#edf4f6')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--metax-audit', type=Path)
    args = ap.parse_args()
    repo = Path(__file__).resolve().parents[1]
    tian = json.loads((repo/'results/2026-11-16-v2-flagtree07-complete/independent-result.json').read_text(encoding='utf-8'))
    assert tian['overall_pass']
    metax = json.loads(args.metax_audit.read_text(encoding='utf-8')) if args.metax_audit else None
    if metax is not None:
        assert metax['workloads_complete'] and metax['source_hashes_match']
    passed = bool(metax and metax['overall_pass'])
    pdfmetrics.registerFont(TTFont('Song', 'C:/Windows/Fonts/simsun.ttc', subfontIndex=0))
    body = ParagraphStyle('body', fontName='Song', fontSize=10.4, leading=16.3, spaceAfter=9, wordWrap='CJK', textColor=NAVY)
    title = ParagraphStyle('title', parent=body, fontSize=24, leading=33, spaceAfter=18)
    head = ParagraphStyle('head', parent=body, fontSize=16, leading=23, spaceAfter=13)
    small = ParagraphStyle('small', parent=body, fontSize=9, leading=14, textColor=GREY)
    flow = []

    def p(text, style=body):
        return Paragraph(html.escape(text).replace('\n', '<br/>'), style)

    def add(text):
        flow.append(p(text))

    def page(title_text):
        if flow:
            flow.append(PageBreak())
        flow.append(p(title_text, head))

    def table(rows, widths):
        cells = [[p(str(v), small) for v in row] for row in rows]
        tab = Table(cells, colWidths=widths, hAlign='LEFT', repeatRows=1)
        tab.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), PALE), ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('LINEBELOW', (0, 0), (-1, 0), .7, TEAL),
            ('LINEBELOW', (0, 1), (-1, -1), .25, colors.HexColor('#d3dfe4')),
            ('LEFTPADDING', (0, 0), (-1, -1), 7), ('RIGHTPADDING', (0, 0), (-1, -1), 7),
            ('TOPPADDING', (0, 0), (-1, -1), 7), ('BOTTOMPADDING', (0, 0), (-1, -1), 5)]))
        flow.extend([tab, Spacer(1, 12)])

    flow.append(p('TokenForge\n面向异构GPU的MiniCPM5-2B吞吐优化', title))
    state = '双平台本地完整验收已通过，待用户审阅' if passed else '审阅草稿：沐曦完整验收尚未通过或仍待完成'
    flow.append(p(state, head))
    add('设计路线：天数采用混合批次Query Tiling，沐曦采用带概率质量认证与回退的Nucleus Shortlist。两个平台沿各自实际热点优化，保留原Attention Backend、模型权重、BF16精度、采样规则与固定服务参数。')
    rows = [['平台 / 结果类型', '4K total tok/s', '16K total tok/s', '本次精度']]
    rows.append(['天数v2 / 11月16日完整验收', '2435.35', '1890.7267', '101/105'])
    if metax:
        rows.append(['沐曦 / 11月17日完整验收', f"{metax['cases'][0]['total_tokens_per_second']:.4f}", f"{metax['cases'][1]['total_tokens_per_second']:.4f}", f"{metax['accuracy_audit']['correct']}/105"])
    else:
        rows.append(['沐曦 / 本次完整验收', '待审计', '待审计', '待审计'])
    table(rows, [190, 102, 105, 110])
    add('完整验收要求：每组原版四轮，丢首轮平均后三轮；吞吐不低于本平台参考99%，Mean TTFT不高于参考101%，新105题完整无执行错误、缺失或空final，精度不低于95%。本报告不混用两卡参考，不拼接会话或借用历史精度。')
    add('贡献归属：受保护的计算分块、全词表排序工作量缩减、成本门禁和可追溯恢复流程属于本作品；主办方FlagTree修复、编译并发限制和HTTP客户端内存修复单独记录，不包装成原创算法收益。')
    flow.append(p('2026-11-17 | 公开作品候选技术报告 | 尚未自动提交比赛平台', small))

    page('1. 天数：混合批次的Query Tiling')
    add('MiniCPM5-2B为BF16、16个Query头/2个KV头、Head Dimension 128。天数采样窗口主要落在原生unified attention；本作品仍调用同一原生Triton kernel，只改变经过验证的Query分块。')
    add('原BLOCK_M=16，候选BLOCK_M=64；保留KV tile、寻址与归约顺序。v1过严的纯prefill门槛遗漏混合批次，v2改用Q_rows >= max(8*num_seqs,128)且Q_rows>num_seqs，同时保留BF16/GQA8/D128、plain causal等特性门禁。纯decode及不支持特性回退原launch。')
    add('核心观察：许多单token decode序列会拉低平均query行数。根据已有形状选择计算分块可扩大有效覆盖，但这不是新的Attention数学算法。共享张量差分与CPU参考检查先于完整模型验收。')
    fig = repo/'docs/figures/20261116-throughput.png'
    if fig.exists():
        flow.append(Image(str(fig), width=500, height=236))
    flow.append(p('图1：原版完整负载的后三轮均值。旧v2与新编译器来自独立会话；编译器修复和环境差异不能全部归因于原创分块代码。', small))
    add('11月16日v2+隔离FlagTree0.7.0完整验收：4K 2435.35tok/s、TTFT5639.32ms；16K 1890.7267tok/s、TTFT268099.60ms；精度101/105，104stop和1max_tokens。旧v2与失败v3原始记录保持不变。')

    page('2. 沐曦：先定位真实执行成本')
    add('C500实际环境：torch2.8.0+metax3.7.0.7、vLLM0.24.0+empty、插件冻结13eb9be69、FlagGems5.3.5、Triton3.0.0+metax3.7.0.7、EvalScope1.11.1。实际词表宽度130560。不把天数厂商补丁直接搬到沐曦。')
    add('12组BF16/GQA decode算子对照未发现通用batch下KV split或头打包的稳定收益；输出缓冲区方案触发原生布局检查，未进入模型。随后在真实模型64请求decode中采样8步，把热点连接到冻结插件实际全词表排序路径。')
    d = Drawing(500, 190)
    for i, (label, val, color) in enumerate([('Sort sweep + histogram', 55.54, TEAL), ('FA2 split KV + combine', 19.58, NAVY), ('Matrix multiplication', 14.39, GREY), ('Other kernels', 10.49, colors.HexColor('#b7c5cf'))]):
        y = 154-i*38
        d.add(String(0, y+6, label, fontName='Helvetica', fontSize=10, fillColor=NAVY))
        d.add(Rect(190, y, val*4.6, 21, fillColor=color, strokeColor=None))
        d.add(String(195+val*4.6, y+6, f'{val:.2f}%', fontName='Helvetica', fontSize=10, fillColor=NAVY))
    flow.append(d)
    flow.append(p('图2：366406.84微秒GPU kernel时长的分类。没有把父Graph/operator时长叠加到子kernel；这是有限decode窗口，不是端到端时间占比或Amdahl收益承诺。', small))
    add('运行问题另记：系统内存64GiB、CPU配额12核但可见128核，默认32个编译进程曾达到62.47GiB后OOM。单编译进程使启动稳定；HTTP benchmark与精度客户端不加载GPU插件，避免重复GPU运行时内存。这些属于环境修复，不算原创优化。')

    page('3. 沐曦：缩减排序范围，保留采样策略')
    add('对top-p-only请求，候选先计算原logits概率与最大值，用单调阈值统计数量/质量。仅当容量4096的集合覆盖nucleus、边界与相等值检查安全时，压紧并排序集合；其他行继续原全排序。保留的logits原值不变，后续RNG实现不变。')
    add('数学依据：集合外较小概率的总质量构成原升序CDF前缀。若集合足够覆盖top-p，只需在集合内定位截断。有限精度中，质量误差、截断边界接近阈值、跨边界相等值均触发保守回退；有限差分不是普遍浮点证明。')
    d = Drawing(500, 178)
    boxes = [(0, 112, '已有CPU调度元数据', 'max_seq_len / batch'), (170, 112, '成本与支持门禁', '未知/长上下文回原生'), (340, 112, '质量认证集合', '最多4096项'), (170, 25, '原全词表过滤', '策略与RNG不变'), (340, 25, '集合排序与边界检查', '不确定则回退')]
    for x, y, a, b in boxes:
        d.add(Rect(x, y, 153, 47, rx=4, ry=4, fillColor=PALE, strokeColor=GREY))
        d.add(String(x+76, y+29, a, fontName='Song', fontSize=9.2, textAnchor='middle'))
        d.add(String(x+76, y+12, b, fontName='Song', fontSize=8.6, textAnchor='middle'))
    for x1, y1, x2, y2 in [(153, 136, 170, 136), (323, 136, 340, 136), (246, 112, 246, 72), (416, 112, 416, 72), (340, 49, 323, 49)]:
        d.add(Line(x1, y1, x2, y2, strokeColor=TEAL, strokeWidth=1.3))
    flow.append(d)
    flow.append(p('图3：现有scheduler max_seq_len更新CPU成本提示，包含Graph replay轮次；不新增读取GPU张量的同步。未知上下文、长度超过8192或batch<64走原过滤。', small))
    add('8192是两个已测成本区间之间的经验保护边界，不是所有中间形状的最优性能证明。门禁只选择两个已检查的过滤实现，不识别benchmark名称、轮次、题目、评分或提示内容，不修改温度/top-p/top-k/输出上限。')

    page('4. 正确性和负结果决定候选取舍')
    table([['诊断版本', '4K吞吐变化', '16K吞吐变化', '结论'], ['静态shortlist', '+8.90%', '-2.11%', '长上下文拒绝'], ['覆盖率EMA成本门禁', '-11.17%', '-1.90%', '拒绝部署'], ['scheduler上下文成本门禁', '+8.85%', '+0.08%', '进入完整验收']], [186, 105, 108, 108])
    flow.append(p('表1：64请求/64并发、1024输出的匹配A/B/A诊断，相对前后原生均值。静态长上下文与后两版本均有形状预热。短测不是原版256/128请求四轮成绩。早先16请求长上下文有67%漂移，其表观收益已废弃。', small))
    add('最终上下文候选的native漂移：4K -0.4137%，16K -0.1822%；TTFT比值1.00329/0.99887；每测量段64请求及65536输出tokens完整。两卡或不同会话的精度从不拼接。')
    add('算子证据：实际词表130560、worker同样FlagGems注册下30组差分，覆盖均匀/稀疏/量化ties/不同top-p/边界；还检查默认与非默认stream。静态首版1040次真实模型shadow逐位相同；上下文集成后1008次逐位相同。Shadow返回原输出，不消耗额外采样随机数，也不作为提速数据。')
    add('本地独立审计必须同时核对源码/完整压缩包hash、实际import、原始CSV及原版summary、预测与review唯一索引0..104、执行错误/空答案、分数与finish reason。exit0、单轮最好成绩或原文件错误overall_pass标记均不能替代审计。')

    page('5. 原版完整验收与本次结果')
    add('固定测试：4K输入/1024输出、并发64、256请求；16K输入/1024输出、并发64、128请求。各四轮，丢首轮平均后三轮。模型、数据、脚本与官方服务参数不变；正式过程没有profiler、developer模式或诊断控制文件。')
    rows = [['负载', '沐曦参考tok/s', '参考TTFT(ms)', '本次tok/s / TTFT(ms)']]
    for i, (label, ref, ttft) in enumerate([('4K', 5089.645, 3199.435), ('16K', 7029.675, 27197.135)]):
        actual = '等待完整审计' if metax is None else f"{metax['cases'][i]['total_tokens_per_second']:.4f} / {metax['cases'][i]['mean_ttft_ms']:.4f}"
        rows.append([label, f'{ref:.3f}', f'{ttft:.3f}', actual])
    table(rows, [50, 125, 126, 206])
    if metax:
        a = metax['accuracy_audit']
        add(f"本次105题：{a['correct']}/105，{a['score_recomputed']:.6%}；finish reason为{a['finish_reasons']}。完整性{a['integrity_pass']}、精度通过{a['accuracy_pass']}、性能通过{metax['performance_pass']}、整体通过{metax['overall_pass']}。原始评分不人工加分、不挑分重跑。")
    else:
        add('本次新105题及完整性能尚待结束，不能填入短测或旧沐曦101/105。若已完成的4K四轮明确失败，可停止该partial session保留预算，但必须标为未完成八轮，不能跨版本拼成绩。')
    add('精度协议：MATH-500 Level3 105题，temperature1.0、top_p0.95、max_tokens32768、batch8、timeout3600，openai_api。--ignore-errors仍需独立完整性检查。触及输出上限必须保留并报告，不能声称所有题自然停止。')
    add('退出与备份：控制器/独立watchdog仅停止自身进程组；截止后停止模型不等于供应商实例关机。完整ROOT、原始CSV、report/预测/review、环境和源文件持续备份到D盘，完成后对照远端hash并扫描凭证，再正常Git推送核对远端提交。')

    page('6. 完整源码、复现架构与贡献边界')
    add('完整源码含408文件，保留各vendor、LICENSE、setup.py、pyproject.toml和csrc构建脚本。插件基点13eb9be69ecc5b5ca4f79c44e9ee40081eaa1bf0；FlagGems为a7620cc官方tree，未修改其源码。Iluvatar v2文件hash与已通过候选相同；MetaX文件与本次实际加载源逐个核验。')
    add('官方镜像已有冻结fl入口，使用PYTHONPATH指向包内完整源码，打印vllm_fl/flag_gems/Triton实际路径后启动。天数专用FlagTree0.7.0仅用于天数，不能安装到沐曦。setup.py可选C++构建仅支持CUDA；本作品MetaX/Iluvatar计算路径采用Python/Triton与镜像自带扩展，不伪造缺失的root CMakeLists。')
    add('跨系统恢复修复：冻结仓库一处NVIDIA MoE JSON符号链接在普通ZIP中成为62字节目标路径文本。当前完整包物化为目标原始3242字节并记录hash。MiniCPM不使用这项MoE配置；这是工程修复，不是性能改动。旧原始档案保留。')
    add('原创贡献：基于真实执行证据的混合批次query分块；保留概率策略的认证式缩小排序范围；受保护的厂商分派和成本回退。它们不是全新Attention数学公式或稀疏近似算法。已有技术思想与本平台实现的创新程度由专家评审，不能保证名次或获奖。')
    add('限制：有限数值/shape/stream检查不是普遍证明；上下文门禁性能仅覆盖实测区间；随机采样会影响精度与执行形态；两个平台成绩来自各自独立会话。编译器和运行环境修复不归为原创收益。正式作品只在用户审阅后上传，当前没有自动提交。')
    flow.append(p('证据索引：results/2026-11-16-v2-flagtree07-complete；results/2026-11-17-metax-diagnostics；experiments/metax_nucleus_context。最终包同时提供readme.md、report.pdf、完整vllm-plugin-FL/及逐文件hash；FlagGems未改，无需伪报Gems优化。', small))

    def footer(canvas, doc):
        canvas.setFont('Song', 8)
        canvas.setFillColor(GREY)
        canvas.drawString(44, 24, 'TokenForge | '+('LOCAL AUDIT PASSED' if passed else 'REVIEW DRAFT - METAX PENDING')+' | 2026-11-17')
        canvas.drawRightString(A4[0]-44, 24, str(doc.page))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    SimpleDocTemplate(str(args.out), pagesize=A4, leftMargin=44, rightMargin=44,
                      topMargin=40, bottomMargin=43, title='TokenForge heterogeneous GPU throughput optimization',
                      author='TokenForge').build(flow, onFirstPage=footer, onLaterPages=footer)
    print(json.dumps(dict(pdf=str(args.out),metax_overall_pass=passed,competition_submitted=False)))


if __name__ == '__main__':
    main()
