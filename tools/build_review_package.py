"""Assemble a credential-free review package; never submits it to a platform."""
import hashlib
import argparse
import html
import json
from pathlib import Path
import shutil
import zipfile
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_LEFT
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Image
from reportlab.graphics.shapes import Drawing, Rect, String, Line
from reportlab.lib.pagesizes import A4


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--validated',action='store_true');parser.add_argument('--out',type=Path);args=parser.parse_args()
    repo=Path(__file__).resolve().parents[1]
    out=args.out or repo.parent/('releases/TokenForge-v2-flagtree070-20261116' if args.validated else 'releases/TokenForge-v2-review-20261116-r3')
    team=out/'TokenForge';team.mkdir(parents=True,exist_ok=False)
    with zipfile.ZipFile(repo/'artifacts/FlagOS-Iluvatar-v2-full-source.zip') as archive:
        assert archive.testzip() is None
        for name in archive.namelist():
            if name.startswith('vllm-plugin-FL/'):
                archive.extract(name,team)
    evidence=team/'evidence';evidence.mkdir()
    shutil.copytree(repo/'results/2026-11-16-accuracy-forensics',evidence/'accuracy-forensics')
    shutil.copy2(repo/'artifacts/FlagOS-Iluvatar-mixed-v2-20261031.zip',evidence)
    shutil.copy2(repo/'results/2026-10-31-iluvatar-mixed-v2/independent-result.json',evidence/'v2-independent-result.json')
    shutil.copy2(repo/'docs/competition-intelligence.md',evidence)
    report_name='submission-report-v2-flagtree070.md' if args.validated else 'submission-report-v2.md'
    readme_name='submission-readme-v2-flagtree070.md' if args.validated else 'submission-readme-v2.md'
    shutil.copy2(repo/'docs'/report_name,team/'report-source.md')
    readme=(repo/'docs'/readme_name).read_text(encoding='utf-8')
    prefix='''# TokenForge - 可审查草稿，尚未正式提交

候选是已完整通过一次验收的天数v2，不采用精度失败的v3。
10/31的完整分数使用旧FlagTree；11/16新编译器实验尚待结束，不能把结果拼接。
report.pdf为当前技术报告草稿，evidence中包含原始v2完整记录和v3评分争议。
正式提交前：更新新编译器结论、复核恢复流程和文档、用户审阅。
此目录没有模型权重、数据集、凭证、Git认证或编译缓存。

以下恢复指南中的仓库相对路径指公开项目内路径；本包完整源码就在vllm-plugin-FL/。
源码实际加载时仍须核对import路径。不要覆盖已验证环境或重复跑基线。

'''
    (team/'readme.md').write_text(readme,encoding='utf-8')
    (team/'tools').mkdir()
    shutil.copy2(repo/'tools/validate_accuracy_artifacts.py',team/'tools')
    shutil.copytree(repo/'results/2026-11-16-compiler-repair',evidence/'compiler-repair',ignore=shutil.ignore_patterns('*.zip'))
    if args.validated:
        completed=repo/'results/2026-11-16-v2-flagtree07-complete'
        check=json.loads((completed/'independent-result.json').read_text(encoding='utf-8'))
        assert check['overall_pass'] and check['accuracy_audit']['correct']==101
        shutil.copytree(completed,evidence/'v2-flagtree070-complete')
        shutil.copy2(repo/'artifacts/FlagOS-Iluvatar-v2-flagtree070-full-20261116.zip',evidence)
        (team/'figures').mkdir()
        for image in (repo/'docs/figures').glob('20261116-*'):shutil.copy2(image,team/'figures'/image.name)
    pdfmetrics.registerFont(TTFont('Song','C:/Windows/Fonts/simsun.ttc',subfontIndex=0))
    body=ParagraphStyle('body',fontName='Song',fontSize=10.5 if args.validated else 11.3,leading=17.0 if args.validated else 18.3,wordWrap='CJK',spaceAfter=10,textColor=colors.HexColor('#233044'))
    title=ParagraphStyle('title',parent=body,fontSize=21,leading=29,spaceAfter=20,textColor=colors.HexColor('#153c57'))
    heading=ParagraphStyle('heading',parent=body,fontSize=15,leading=22,spaceAfter=14,spaceBefore=12,keepWithNext=True,textColor=colors.HexColor('#153c57'))
    source=(repo/'docs'/report_name).read_text(encoding='utf-8')
    flow=[]
    for block in source.split('\n\n'):
        text=block.strip()
        if not text:continue
        if text.startswith('## '):
            if text.startswith(('## 3.', '## 5.')):flow.append(PageBreak())
            flow.append(Paragraph(html.escape(text[3:]),heading))
        elif text.startswith('# '):flow.append(Paragraph(html.escape(text[2:]),title))
        else:flow.append(Paragraph(html.escape(text).replace('\n','<br/>'),body))
    if args.validated:
        flow.extend([PageBreak(),Paragraph('7. 混合批次的分派路径与性能证据',heading)])
        graph=Drawing(500,125)
        texts=[['已有batch元数据','Q行数 / 序列数'],['特性及形状门禁','BF16 / GQA8 / D128'],['满足：M64 / Q8','否则：原生launch'],['同一原生Triton内核','KV寻址及归约不变']]
        for i,lines in enumerate(texts):
            xx=5+i*125
            graph.add(Rect(xx,60,112,49,rx=4,ry=4,strokeColor=colors.HexColor('#65859a'),fillColor=colors.HexColor('#edf4f7')))
            for j,line in enumerate(lines):graph.add(String(xx+56,91-j*16,line,fontName='Song',fontSize=9.2,textAnchor='middle'))
            if i<3:
                graph.add(Line(xx+113,84,xx+124,84,strokeColor=colors.HexColor('#65859a')))
                graph.add(Line(xx+120,87,xx+124,84));graph.add(Line(xx+120,81,xx+124,84))
        graph.add(String(6,36,'形状门槛：Q_rows >= max(8 * num_seqs, 128)，且Q_rows > num_seqs。',fontName='Song',fontSize=10))
        graph.add(String(6,17,'非支持特性与纯decode回退；不按benchmark名称、题目或数据内容分派。',fontName='Song',fontSize=10))
        flow.append(graph)
        flow.append(Paragraph('图1：同一native kernel的受保护launch adapter。核心观察是混合batch中大量单token decode会拉低平均query行数，固定按纯prefill门槛会漏掉可获益形状；分派只使用已有公开形状信息。它不构成新的Attention算法。',body))
        flow.append(Image(str(repo/'docs/figures/20261116-throughput.png'),width=500,height=236))
        flow.append(Paragraph('图2：完整八轮的后三轮均值（柱顶为total tok/s）。相对公开参考的性能差距可观；旧v2与新环境来自不同会话，不能把编译器带来的额外收益全部归因原创优化。',body))
        flow.extend([PageBreak(),Paragraph('8. 正确性证据、创新归属与剩余缺口',heading)])
        flow.append(Image(str(repo/'docs/figures/20261116-compiler-error.png'),width=500,height=243))
        flow.append(Paragraph('图3：全16组相同输入上所选行的CPU参考最大绝对误差。新旧源码、Q/K/V和分页表不变；新版本全部通过既定atol/rtol，未放宽标准。标量fsum另证CPU参考一致。它是编译器修复的证据，不证明完整模型所有logits或随机采样因果。',body))
        flow.append(Paragraph('可归属本作品的贡献：面向BF16/GQA混合批次的受保护query分块、原生路径保留及逐级门禁验证。主办方提供的FlagTree更新不能计为我们的原创；评分器法证和完整归档是工程可信度，不等同新算法。创新分最终由专家判断。',body))
        flow.append(Paragraph('当前最大缺口是沐曦：原始4K未达标，prefix元数据跨层复用仅有短测观察，没有本轮完整官方验收，因此本包不宣称沐曦优化通过。官网总分包含两卡性能与创新分，且公开榜单尚无成绩；无法承诺获奖或排名。单平台未过基础门槛的最终有效性应由赛方确认。',body))
    def page(canvas,doc):
        canvas.setFont('Song',8);canvas.setFillColor(colors.HexColor('#6a7b8b'))
        canvas.drawString(46,25,'TokenForge | V2 '+('LOCAL VALIDATION PASSED' if args.validated else 'REVIEW DRAFT')+' | 2026-11-16')
        canvas.drawRightString(A4[0]-46,25,str(doc.page))
    SimpleDocTemplate(str(team/'report.pdf'),pagesize=A4,rightMargin=46,leftMargin=46,topMargin=44,bottomMargin=44,
        title='TokenForge MiniCPM5-2B query tiling v2 - review draft',author='TokenForge').build(flow,onFirstPage=page,onLaterPages=page)
    files={str(p.relative_to(team)).replace('\\','/'):hashlib.sha256(p.read_bytes()).hexdigest() for p in team.rglob('*') if p.is_file()}
    assert not any('/private/' in n or n.startswith('AGENTS.md') for n in files)
    manifest=dict(user_review_required=True,competition_submitted=False,compiler_validation_pending=not args.validated,local_validation_passed=args.validated,files=files)
    (team/'REVIEW_MANIFEST.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    archive=out/('TokenForge.zip' if args.validated else 'TokenForge-review.zip')
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(team.rglob('*')):
            if p.is_file():z.write(p,'TokenForge/'+p.relative_to(team).as_posix())
    with zipfile.ZipFile(archive) as z:assert z.testzip() is None
    print(json.dumps(dict(package=str(archive),sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),files=len(files),submitted=False)))


if __name__=='__main__':main()
