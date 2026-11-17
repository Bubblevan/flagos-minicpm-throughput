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
    add('设计路线：天数采用混合批次Query Tiling，沐曦采用受保护的原生NT矩阵乘Launch Geometry。两个平台沿各自实际热点优化，保留原Attention Backend、模型权重、BF16精度、采样规则与固定服务参数。')
    rows = [['平台 / 结果类型', '4K total tok/s', '16K total tok/s', '本次精度']]
    rows.append(['天数v2 / 11月16日完整验收', '2435.35', '1890.7267', '101/105'])
    if metax:
        rows.append(['沐曦 / 11月17日完整验收', f"{metax['cases'][0]['total_tokens_per_second']:.4f}", f"{metax['cases'][1]['total_tokens_per_second']:.4f}", f"{metax['accuracy_audit']['correct']}/105"])
    else:
        rows.append(['沐曦 / 本次完整验收', '待审计', '待审计', '待审计'])
    table(rows, [190, 102, 105, 110])
    add('完整验收要求：每组原版四轮，丢首轮平均后三轮；吞吐不低于本平台参考99%，Mean TTFT不高于参考101%，新105题完整无执行错误、缺失或空final，精度不低于95%。本报告不混用两卡参考，不拼接会话或借用历史精度。')
    add('贡献归属：受保护的混合Query分块、形状感知矩阵乘分块和可追溯恢复流程属于本作品；主办方FlagTree修复、编译并发限制和HTTP客户端内存修复单独记录，不包装成原创算法收益。')
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

    page('2. 沐曦：不同执行窗口有不同热点')
    add('C500环境：torch2.8.0+metax3.7.0.7、vLLM0.24.0+empty、冻结插件13eb9be69、FlagGems5.3.5/a762、Triton3.0.0+metax3.7.0.7、EvalScope1.11.1。不能把天数厂商补丁直接套到沐曦。')
    add('先检查实际MacaFlashAttention路径，KV split与头打包没有稳定收益。随后decode窗口指向全词表排序；但新的真实prefill窗口显示NT矩阵乘占73.29%。两个窗口不矛盾，不能把局部占比当作整个端到端耗时。')
    d = Drawing(500, 180)
    palette=[TEAL,NAVY,GREY,colors.HexColor('#b7c5cf')]
    for y,label,vals in [(112,'Decode: 8 steps',[14.39,55.54,19.58,10.49]),(48,'Prefill: 8 steps',[73.2852,2.5824,15.1019,9.0305])]:
        d.add(String(0,y+28,label,fontName='Helvetica',fontSize=10,fillColor=NAVY))
        x=0
        for v,c in zip(vals,palette):
            d.add(Rect(x,y,v*5,21,fillColor=c,strokeColor=None));x+=v*5
    for i,(name,c) in enumerate(zip(['NT GEMM','Sort','Attention','Other'],palette)):
        d.add(Rect(i*122,9,11,11,fillColor=c,strokeColor=None));d.add(String(i*122+16,10,name,fontName='Helvetica',fontSize=9,fillColor=NAVY))
    flow.append(d)
    flow.append(p('图2：GPU子kernel时长分类，未叠加父Graph/operator。Decode窗口366406.84us；prefill窗口872568.832us，1344次NT GEMM=8步*42层*4矩阵乘。不是全程剖析或收益上界承诺。',small))
    add('运行修复单独归属：64GiB系统内存、12核配额却可见128核，原32个编译进程曾导致OOM。改为单编译进程，HTTP测试客户端不加载GPU插件以避免重复运行时。11/17傍晚的新容器另出现原生Torch设备队列超时；这不是分块优化收益或精度结论。')

    page('3. 沐曦：保留原计算，选择可验证分块')
    add('实际prefill块M=2048，BF16连续A、转置连续权重B、连续输出C。候选复用原生mm_kernel_nt的完整计算AST，只对三类经过测量的(N,K)选择Launch Geometry；O投影无稳定收益，保留原路径。')
    table([['用途 / (N,K)','候选(BM,BN,BK)','微测耗时(ms)'],['QKV / (2560,2048)','(256,128,32)','0.24407 -> 0.21127'],['Gate+Up / (12288,2048)','(128,256,32)','0.91506 -> 0.86650'],['Down / (2048,6144)','(128,128,64)','0.57396 -> 0.47295'],['O / (2048,2048)','保留原生','差异不具实际意义']], [190,150,167])
    add('三种候选均使用8warps、2stages、basic pipeline；GROUP_M仍为8，float32累加、allow_tf32=False、BF16输出。相同计算AST不等于所有浮点归约结果普遍相同，尤其改变K分块时仍须数值验证。')
    add('保护条件只依赖张量形状、dtype与layout：M必须2048，且只支持上表三组N/K。其他M、decode、不支持的dtype/layout、O投影均保留原tuner与launch。没有识别benchmark、轮次、提示或评分，也未改服务超参、模型权重和采样规则。')
    add('8组分块穷举微测中，仅从逐位原生一致且独立CPU参考通过的候选里选配置。四类矩阵乘加权GPU耗时减少9.54%，只是算子诊断，不等于端到端吞吐提升。核心insight是按实际矩阵形状改善原生计算的分块，而不是更换backend或稀疏近似。')

    page('4. 数值、真实负载与反例约束')
    add('隔离完整FlagGems树的实际public函数通过18项测试：6种形状*随机/抵消/零输入，含不优化的O投影与M127/64回退。全部输出与原生逐位一致；选取行另以CPU float64矩阵乘核对，未放宽已有容差。这是有限证据，不是普遍数值证明。')
    table([['实验','结果','决策'],['静态/自适应shortlist','长上下文或两组回退','拒绝'],['上下文shortlist完整4K','4807.27tok/s / 3471.45ms','失败；无16K/新105'],['融合shortlist充分预热','-0.0454%端到端收益','拒绝'],['NT GEMM 原生A1','5021.24tok/s / 5549.11ms','匹配诊断'],['NT GEMM 候选B','5177.19tok/s / 5127.27ms','吞吐+3.38%；TTFT-8.57%'],['NT GEMM 原生A2','4994.61tok/s / 5666.08ms','原生漂移-0.53%']], [165,225,117])
    flow.append(p('表2：GEMM三段分别重启服务，每段64请求预热、128请求测量，4096输入/1024输出、并发64、原随机文本生成器。请求和token完整。不能直接与官方256请求TTFT或四轮均值混比。',small))
    add('预注册门禁为吞吐>=3%提升、原生漂移<=3%、TTFT不回退，本次诊断通过。原生采样与prefix元数据补丁在三段中保持相同，仅替换FlagGems隔离源码。三个生成kernel与实际import路径、源文件hash均有记录。')
    add('冷首段对比的采样+14.6%已因14.92%原生漂移作废；不以此挑分或挑最好一轮。15:52容器中断的首次GEMM正式尝试没有原始CSV，也未进入新105题，不能算完整评测。原始证据保留，不跨中断拼接。')
    add('新64GiB机器18项数值复核通过后，GEMM＋prefix组合四轮4K为4504.03tok/s、TTFT3288.36ms，两项失败。现恢复冻结原生MetaX Attention，仅保留GEMM候选进行独立完整验收；这是隔离变量，未证明prefix导致回退。历史A/B/A不能冒充本次GEMM-only端到端收益。')

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
    add('提交候选同时提供完整vllm-plugin-FL与修改后的FlagGems源码、LICENSE和编译/安装脚本。插件基点13eb9be69，Gems基点a7620cc；Gems仅修改MetaX NT GEMM。双平台集成源码5186文件，逐文件manifest和ZIP SHA256均可独立核验。')
    add('当前GEMM-only源码已将MetaX Attention恢复冻结原版，原生sampler不变。完整源码ZIP哈希c3d8ad41789bbbf64f30e47ad291cf6ffdc5c207d7d0e52b873659e1d3895146。上一组合失败源码与完整四轮反例另行保留，不作为提交有效成绩。')
    add('保护天数：platform.py与attention_tiles.py精确取自已通过v2完整源码；相较C500测试树，共同platform仅新增vendor_name==iluvatar分支，MetaX条件为false，其他MetaX及Gems字节不变。静态集成证据不冒充双平台新运行；仍须核对最终加载树与验收会话。')
    add('镜像已有冻结fl入口，PYTHONPATH指向包内完整插件及FlagGems/src；打印实际import、包版本、硬件与启动argv。天数FlagTree0.7.0仅用于天数。C500单编译进程和CPU-only HTTP客户端均单独标注为运行修复，不能当作原创算子收益。')
    add('完整包物化上游NVIDIA MoE JSON链接为目标原始3242字节，保留hash；MiniCPM不走MoE，这只是ZIP恢复修复。源码/cache置于证据ROOT外，持续备份原CSV、report、预测、review、源码manifest与日志；最终核对远端hash再推送。')
    add('原创贡献聚焦真实执行热点驱动的混合Query分块、受保护的MetaX形状感知NT GEMM配置以及完整恢复和独立审计流程。不是新Attention数学公式，不声称稀疏算法或普遍无损；排名和创新程度由评审判断，不能保证获奖。')
    add('限制：有限shape/数值测试不等于普遍证明；局部profile不等于全程；随机采样与跨会话环境变化可能影响分数。设备故障、主办方编译器修复不包装成原创优化。正式作品须经用户审阅，当前未自动上传比赛平台。')
    flow.append(p('证据索引：results/2026-11-16-v2-flagtree07-complete；results/2026-11-17-metax-diagnostics；experiments/metax_prefill_mm。最终结构必须含readme.md、report.pdf、vllm-plugin-FL/、FlagGems/；不含凭证、模型权重、数据集或编译缓存。',small))

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
