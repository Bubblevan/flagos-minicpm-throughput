"""Assemble a credential-free review package; never submits it to a platform."""
import hashlib
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
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
from reportlab.lib.pagesizes import A4


def main():
    repo=Path(__file__).resolve().parents[1]
    out=repo.parent/'releases/TokenForge-v2-review-20261116-r3'
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
    shutil.copy2(repo/'docs/submission-report-v2.md',team/'report-source.md')
    readme=(repo/'docs/submission-readme-v2.md').read_text(encoding='utf-8')
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
    pdfmetrics.registerFont(TTFont('Song','C:/Windows/Fonts/simsun.ttc',subfontIndex=0))
    body=ParagraphStyle('body',fontName='Song',fontSize=11.3,leading=18.3,wordWrap='CJK',spaceAfter=10,textColor=colors.HexColor('#233044'))
    title=ParagraphStyle('title',parent=body,fontSize=21,leading=29,spaceAfter=20,textColor=colors.HexColor('#153c57'))
    heading=ParagraphStyle('heading',parent=body,fontSize=15,leading=22,spaceAfter=14,spaceBefore=12,keepWithNext=True,textColor=colors.HexColor('#153c57'))
    source=(repo/'docs/submission-report-v2.md').read_text(encoding='utf-8')
    flow=[]
    for block in source.split('\n\n'):
        text=block.strip()
        if not text:continue
        if text.startswith('## '):
            if text.startswith(('## 3.', '## 5.')):flow.append(PageBreak())
            flow.append(Paragraph(html.escape(text[3:]),heading))
        elif text.startswith('# '):flow.append(Paragraph(html.escape(text[2:]),title))
        else:flow.append(Paragraph(html.escape(text).replace('\n','<br/>'),body))
    def page(canvas,doc):
        canvas.setFont('Song',8);canvas.setFillColor(colors.HexColor('#6a7b8b'))
        canvas.drawString(46,25,'TokenForge | V2 REVIEW DRAFT | 2026-11-16')
        canvas.drawRightString(A4[0]-46,25,str(doc.page))
    SimpleDocTemplate(str(team/'report.pdf'),pagesize=A4,rightMargin=46,leftMargin=46,topMargin=44,bottomMargin=44,
        title='TokenForge MiniCPM5-2B query tiling v2 - review draft',author='TokenForge').build(flow,onFirstPage=page,onLaterPages=page)
    files={str(p.relative_to(team)).replace('\\','/'):hashlib.sha256(p.read_bytes()).hexdigest() for p in team.rglob('*') if p.is_file()}
    assert not any('/private/' in n or n.startswith('AGENTS.md') for n in files)
    manifest=dict(review_only=True,competition_submitted=False,compiler_validation_pending=True,files=files)
    (team/'REVIEW_MANIFEST.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    archive=out/'TokenForge-review.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(team.rglob('*')):
            if p.is_file():z.write(p,'TokenForge/'+p.relative_to(team).as_posix())
    with zipfile.ZipFile(archive) as z:assert z.testzip() is None
    print(json.dumps(dict(package=str(archive),sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),files=len(files),submitted=False)))


if __name__=='__main__':main()
