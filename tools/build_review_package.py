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
    parser=argparse.ArgumentParser();parser.add_argument('--validated',action='store_true');parser.add_argument('--final',action='store_true');parser.add_argument('--out',type=Path);args=parser.parse_args()
    if args.final: args.validated=True
    repo=Path(__file__).resolve().parents[1]
    out=args.out or repo.parent/('releases/TokenForge-v2-flagtree070-20261116' if args.validated else 'releases/TokenForge-v2-review-20261116-r3')
    team=out/'TokenForge';team.mkdir(parents=True,exist_ok=False)
    with zipfile.ZipFile(repo/'artifacts/FlagOS-Iluvatar-v2-full-source.zip') as archive:
        assert archive.testzip() is None
        if args.final:
            (team/'SOURCE_MANIFEST.json').write_bytes(archive.read('SOURCE_MANIFEST.json'))
        for name in archive.namelist():
            if name.startswith('vllm-plugin-FL/'):
                archive.extract(name,team)
    evidence=team/'evidence';evidence.mkdir()
    shutil.copytree(repo/'results/2026-11-16-accuracy-forensics',evidence/'accuracy-forensics')
    shutil.copy2(repo/'artifacts/FlagOS-Iluvatar-mixed-v2-20261031.zip',evidence)
    shutil.copy2(repo/'results/2026-10-31-iluvatar-mixed-v2/independent-result.json',evidence/'v2-independent-result.json')
    shutil.copy2(repo/'docs/competition-intelligence.md',evidence)
    report_name='submission-report-final-20261118-r2.md' if args.final else ('submission-report-v2-flagtree070.md' if args.validated else 'submission-report-v2.md')
    readme_name='submission-readme-final-20261118.md' if args.final else ('submission-readme-v2-flagtree070.md' if args.validated else 'submission-readme-v2.md')
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
    if args.final:
        for name in ['audit_submission.py','finalize_saved_result.py','build_iluvatar_candidate.py']:
            shutil.copy2(repo/'tools'/name,team/'tools'/name)
        shutil.copy2(repo/'docs/metax-closeout-20261118.md',evidence/'metax-closeout.md')
        shutil.copy2(repo/'artifacts/FlagOS-MetaX-baseline-20261029.zip',evidence)
        shutil.copy2(repo/'artifacts/FlagOS-MetaX-NN-dynamic-ABA-rejected-20261118.zip',evidence)
        shutil.copytree(repo/'results/2026-10-29-metax-baseline',evidence/'metax-baseline')
    shutil.copytree(repo/'results/2026-11-16-compiler-repair',evidence/'compiler-repair',ignore=shutil.ignore_patterns('*.zip'))
    if args.validated:
        completed=repo/'results/2026-11-16-v2-flagtree07-complete'
        check=json.loads((completed/'independent-result.json').read_text(encoding='utf-8'))
        assert check['overall_pass'] and check['accuracy_audit']['correct']==101
        shutil.copytree(completed,evidence/'v2-flagtree070-complete')
        shutil.copy2(repo/'artifacts/FlagOS-Iluvatar-v2-flagtree070-full-20261116.zip',evidence)
        (team/'figures').mkdir()
        for image in (repo/'docs/figures').glob(('20261118' if args.final else '20261116')+'-*'):shutil.copy2(image,team/'figures'/image.name)
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
        graph.add(String(6,17,'分派采用已有shape及特性元数据；其余特性与纯decode使用原生launch。',fontName='Song',fontSize=10))
        flow.append(graph)
        flow.append(Paragraph('图1：复用native kernel的launch adapter。混合batch中的单token decode拉低平均query行数；形状感知门槛提升适用混合批次的优化覆盖，原生回退路径处理其余调用。',body))
        figure_prefix='20261118' if args.final else '20261116'
        flow.append(Image(str(repo/f'docs/figures/{figure_prefix}-throughput.png'),width=500,height=236))
        flow.append(Paragraph('图2：官方参考与两种编译器环境下v2的完整实测结果，柱顶标注total tok/s。每次实验均按原版四轮丢首轮、平均后三轮统计。新编译器环境采用主办方推荐的FlagTree0.7.0。',body))
        flow.extend([PageBreak(),Paragraph('8. 数值验证与技术贡献',heading)])
        flow.append(Image(str(repo/f'docs/figures/{figure_prefix}-compiler-error.png'),width=500,height=243))
        flow.append(Paragraph('图3：16组共享输入的CPU参考最大绝对误差。实验保持Attention源码、Q/K/V与分页表相同，对比两种编译器环境。FlagTree0.7.0全部通过固定atol=0.02、rtol=0.02检查；标量fsum交叉核验支持CPU参考一致。',body))
        flow.append(Paragraph('本作品的技术贡献包括：混合batch结构分析、BF16/GQA形状感知query分块、原生回退集成，以及从算子到模型的分层验证。编译器环境采用主办方发布的FlagTree修复；完整原始记录与离线审计共同支撑复现流程。',body))
        metax_text=('天数v2在4K和16K场景均达到性能门槛，MATH-500 Level3为101/105。沐曦原始4K未达标，探索已结束，最终源码保留MetaX冻结原生路径。分平台结果、原始记录和实验决策附于evidence/。' if args.final else '沐曦原始4K未达标，最终源码保留MetaX原生路径；天数完整实测结果与精度记录分别归档。')
        flow.append(Paragraph(metax_text,body))
    def page(canvas,doc):
        canvas.setFont('Song',8);canvas.setFillColor(colors.HexColor('#6a7b8b'))
        canvas.drawString(46,25,'TokenForge | V2 '+('ILUVATAR LOCAL VALIDATION PASSED' if args.final else ('LOCAL VALIDATION PASSED' if args.validated else 'REVIEW DRAFT'))+(' | 2026-11-18' if args.final else ' | 2026-11-16'))
        canvas.drawRightString(A4[0]-46,25,str(doc.page))
    SimpleDocTemplate(str(team/'report.pdf'),pagesize=A4,rightMargin=46,leftMargin=46,topMargin=44,bottomMargin=44,
        title='TokenForge MiniCPM5-2B query tiling v2'+(' - final candidate' if args.final else ' - review draft'),author='TokenForge').build(flow,onFirstPage=page,onLaterPages=page)
    files={str(p.relative_to(team)).replace('\\','/'):hashlib.sha256(p.read_bytes()).hexdigest() for p in team.rglob('*') if p.is_file()}
    assert not any('/private/' in n or n.startswith('AGENTS.md') for n in files)
    manifest=dict(user_review_required=True,competition_submitted=False,compiler_validation_pending=not args.validated,local_validation_passed=args.validated,files=files)
    if args.final: manifest.update(declared_optimized_platforms=['iluvatar'],metax_optimization_passed=False,release_date='2026-11-18',source_unchanged_from_validated_v2=True)
    (team/'REVIEW_MANIFEST.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    archive=out/('TokenForge.zip' if args.validated else 'TokenForge-review.zip')
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(team.rglob('*')):
            if p.is_file():z.write(p,'TokenForge/'+p.relative_to(team).as_posix())
    with zipfile.ZipFile(archive) as z:assert z.testzip() is None
    print(json.dumps(dict(package=str(archive),sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),files=len(files),submitted=False)))


if __name__=='__main__':main()
